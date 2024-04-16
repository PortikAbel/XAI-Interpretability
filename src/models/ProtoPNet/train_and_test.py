import argparse
import time

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import f1_score

from models.ProtoPNet.util.helpers import list_of_distances
from utils.log import Log


def _train_or_test(
    args: argparse.Namespace,
    epoch: int,
    model: torch.nn.DataParallel | torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    log: Log,
    optimizer: torch.optim.Optimizer = None,
    class_specific: bool = True,
    use_l1_mask: bool = True,
    tensorboard_writer=None,
) -> float:
    """
    Perform a train or test step.

    :param args:
    :param epoch: current epoch number
    :param model: multi-gpu model
    :param dataloader:
    :param log: logger
    :param optimizer: if ``None``, then no gradient update is performed.
        Defaults to ``None``.
    :param class_specific: Defaults to ``True``.
    :param use_l1_mask: Defaults to ``True``.
    :param tensorboard_writer: Defaults to ``None``.
    :return: achieved accuracy
    """
    is_train = optimizer is not None
    is_pretrain = epoch < args.epochs_warm
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0

    true_labels = np.array([])
    predicted_labels = np.array([])

    total_steps = len(dataloader) * (epoch - 1)  # epoch number starts from 1
    for current_step, (image, label) in enumerate(dataloader, start=1):
        input_ = image.cuda()
        target_ = label.cuda()
        true_labels = np.append(true_labels, label.numpy())

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, additional_out = model(input_)

            loss, cross_entropy, cluster_cost, l1, l2, separation_cost = (
                compute_loss_components(
                    input_=input_,
                    target_=target_,
                    label=label,
                    model=model,
                    optimizer=optimizer,
                    output=output,
                    additional_out=additional_out,
                    args=args,
                    class_specific=class_specific,
                    use_l1_mask=use_l1_mask,
                    tensorboard_writer=tensorboard_writer,
                    step=total_steps + current_step,
                    is_train=is_train,
                    is_pretrain=is_pretrain,
                )
            )

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target_.size(0)
            n_correct += (predicted == target_).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

        del input_
        del target_
        del output
        del predicted

    end = time.time()

    log.info(f"\t\t{'time:':13}{end - start}")
    log.info(f"\t\t{'cross ent:':13}{total_cross_entropy / n_batches}")
    log.info(f"\t\t{'cluster:':13}{total_cluster_cost / n_batches}")
    if class_specific:
        log.info(f"\t\t{'separation:':13}{total_separation_cost / n_batches}")
    log.info(f"\t\t{'accu:':13}{n_correct / n_examples:.2%}")
    log.info(
        f"\t\t{'micro f1:':13}"
        f"{f1_score(true_labels, predicted_labels, average='micro')}"
    )
    log.info(
        f"\t\t{'macro f1:':13}"
        f"{f1_score(true_labels, predicted_labels, average='macro')}"
    )
    log.info(f"\t\t{'l1:':13}{model.module.last_layer.weight.norm(p=1).item()}")
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log.info(f"\t\tp dist pair: {p_avg_pair_dist.item()}")

    return n_correct / n_examples


def compute_loss_components(
    input_,
    target_,
    label,
    model,
    optimizer,
    output,
    additional_out,
    args,
    class_specific,
    use_l1_mask,
    tensorboard_writer,
    step,
    is_train,
    is_pretrain,
):
    _, predicted = torch.max(output.data, 1)
    n_correct = (predicted == target_).sum().item()
    acc = n_correct / len(predicted)

    # compute loss
    if args.binary_cross_entropy:
        one_hot_target = torch.nn.functional.one_hot(target_, args.num_classes)
        cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
            output, one_hot_target.float(), reduction="sum"
        )
    else:
        cross_entropy = torch.nn.functional.cross_entropy(output, target_)

    l2 = torch.tensor(0.0)
    separation_cost = torch.tensor(0.0)
    if class_specific:
        max_dist = (
            model.module.prototype_shape[1]
            * model.module.prototype_shape[2]
            * model.module.prototype_shape[3]
        )

        # prototypes_of_correct_class is a tensor
        # of shape batch_size * num_prototypes
        # calculate cluster cost
        prototypes_of_correct_class = torch.t(
            model.module.prototype_class_identity[:, label]
        ).cuda()
        inverted_distances, target_proto_index = torch.max(
            (max_dist - additional_out.min_distances) * prototypes_of_correct_class,
            dim=1,
        )
        cluster_cost = torch.mean(max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class

        if args.separation_type == "max":
            (
                inverted_distances_to_nontarget_prototypes,
                _,
            ) = torch.max(
                (max_dist - additional_out.min_distances) * prototypes_of_wrong_class,
                dim=1,
            )
            separation_cost = torch.mean(
                max_dist - inverted_distances_to_nontarget_prototypes
            )
        elif args.separation_type == "avg":
            min_distances_detached_prototype_vectors = (
                model.module.prototype_min_distances(input_, detach_prototypes=True)[0]
            )
            # calculate avg cluster cost
            avg_separation_cost = torch.sum(
                min_distances_detached_prototype_vectors * prototypes_of_wrong_class,
                dim=1,
            ) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            l2 = (
                torch.mm(
                    model.module.prototype_vectors[:, :, 0, 0],
                    model.module.prototype_vectors[:, :, 0, 0].t(),
                )
                - torch.eye(args.prototype_shape[0]).cuda()
            ).norm(p=2)

            separation_cost = avg_separation_cost
        elif args.separation_type == "margin":
            # For each input get the distance
            # to the closest target class prototype
            min_distance_target = max_dist - inverted_distances.reshape((-1, 1))

            all_distances = additional_out.distances
            min_indices = additional_out.min_indices

            anchor_index = min_indices[
                torch.arange(0, target_proto_index.size(0), dtype=torch.long),
                target_proto_index,
            ].squeeze()
            all_distances = all_distances.view(
                all_distances.size(0), all_distances.size(1), -1
            )
            distance_at_anchor = all_distances[
                torch.arange(0, all_distances.size(0), dtype=torch.long),
                :,
                anchor_index,
            ]

            # For each non-target prototype
            # compute difference compared to the closest target prototype
            # d(a, p) - d(a, n) term from TripletMarginLoss
            distance_pos_neg = (
                min_distance_target - distance_at_anchor
            ) * prototypes_of_wrong_class
            # Separation cost is the margin loss
            # max(d(a, p) - d(a, n) + margin, 0)
            separation_cost = torch.mean(
                torch.maximum(
                    distance_pos_neg + args.coefs["sep_margin"],
                    torch.tensor(0.0, device=distance_pos_neg.device),
                )
            )
        else:
            raise ValueError(
                f"separation_type has to be one of [max, mean, margin], "
                f"got {args.separation_type}"
            )

        if use_l1_mask:
            l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
        else:
            l1 = model.module.last_layer.weight.norm(p=1)

    else:
        min_distance, _ = torch.min(additional_out.min_distances, dim=1)
        cluster_cost = torch.mean(min_distance)
        l1 = model.module.last_layer.weight.norm(p=1)

    # compute gradient and do SGD step
    loss = None
    if is_train:
        if class_specific:
            loss = (
                args.coefs["crs_ent"] * cross_entropy
                + args.coefs["clst"] * cluster_cost
                + args.coefs["sep"] * separation_cost
                + (args.coefs["l2"] * l2 if args.separation_type == "avg" else 0)
                + args.coefs["l1"] * l1
            )
        else:
            loss = (
                args.coefs["crs_ent"] * cross_entropy
                + args.coefs["clst"] * cluster_cost
                + args.coefs["l1"] * l1
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if tensorboard_writer:
        if is_train:
            if is_pretrain:
                phase = "pretrain"
            else:
                phase = "train"
        else:
            phase = "test"

        tensorboard_writer.add_scalar(f"Loss/{phase}/loss", loss.item(), step)
        tensorboard_writer.add_scalar(
            f"Loss/{phase}/cross entropy", cross_entropy.item(), step
        )
        tensorboard_writer.add_scalar(
            f"Loss/{phase}/cluster cost", cluster_cost.item(), step
        )
        tensorboard_writer.add_scalar(
            f"Loss/{phase}/separation cost", separation_cost.item(), step
        )
        tensorboard_writer.add_scalar(f"Loss/{phase}/l1", l1.item(), step)
        tensorboard_writer.add_scalar(f"Loss/{phase}/l2", l2.item(), step)
        tensorboard_writer.add_scalar(f"Loss/{phase}/Accuracy", acc, step)

    return loss, cross_entropy, cluster_cost, l1, l2, separation_cost


def train(
    args: argparse.Namespace,
    epoch: int,
    model: torch.nn.DataParallel | torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    log: Log,
    optimizer: torch.optim.Optimizer,
    class_specific: bool = False,
    tensorboard_writer=None,
) -> float:
    """
    Train the model.

    :param args:
    :param epoch: current epoch number
    :param model: multi-gpu model
    :param dataloader:
    :param log:
    :param optimizer:
    :param class_specific: Defaults to ``False``.
    :param tensorboard_writer: Defaults to ``None``.
    :return: train accuracy
    """
    assert optimizer is not None

    log.info("\ttrain")
    model.train()
    return _train_or_test(
        args=args,
        epoch=epoch,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        class_specific=class_specific,
        log=log,
        tensorboard_writer=tensorboard_writer,
    )


def test(
    args: argparse.Namespace,
    epoch: int,
    model: torch.nn.DataParallel | torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    log: Log,
    class_specific: bool = False,
    tensorboard_writer=None,
) -> float:
    """
    Test the model.

    :param args:
    :param epoch: current epoch number
    :param model: multi-gpu model
    :param dataloader:
    :param log:
    :param class_specific: Defaults to ``False``.
    :param tensorboard_writer: Defaults to ``None``.
    :return: test accuracy
    """
    log.info("\ttest")
    model.eval()
    return _train_or_test(
        args=args,
        epoch=epoch,
        model=model,
        dataloader=dataloader,
        optimizer=None,
        class_specific=class_specific,
        log=log,
        tensorboard_writer=tensorboard_writer,
    )


def last_only(model, log):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info("\tlast layer")


def warm_only(model, log):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info("\twarm")


def joint(model, log):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info("\tjoint")
