import argparse
from functools import partial

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.ProtoPNet import model, push
from models.ProtoPNet import train_and_test as tnt
from models.ProtoPNet.util import save
from models.ProtoPNet.util.data import get_dataloaders
from models.ProtoPNet.util.preprocess import preprocess
from utils.log import Log


def _train_backbone_model(log, tensorboard_writer, args):
    (
        train_loader,
        _,
        test_loader,
        classes,
    ) = get_dataloaders(log, args)

    # we should look into distributed sampler more carefully
    # at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log.info(f"training set size: {len(train_loader.dataset)}")
    log.info(f"test set size: {len(test_loader.dataset)}")
    log.info(f"batch size: {args.batch_size}")

    # construct the model
    ppnet = model.construct_PPNet(
        base_architecture=args.net,
        backbone_only=args.backbone_only,
        pretrained=not args.disable_pretrained,
        img_shape=args.img_shape,
        prototype_shape=args.prototype_shape,
        num_classes=args.num_classes,
        prototype_activation_function=args.prototype_activation_function,
        add_on_layers_type=args.add_on_layers_type,
    )
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    joint_optimizer_specs = [
        {
            "params": ppnet.features.parameters(),
            "lr": args.joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": args.joint_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=args.joint_lr_step, gamma=0.1
    )

    # train the model
    log.info("start training")

    epoch = 1
    accu = 0.0
    while epoch <= args.n_epochs:
        log.info(f"epoch: \t{epoch} / {args.n_epochs}")
        tnt.joint(model=ppnet_multi, log=log)
        if epoch > 1:
            joint_lr_scheduler.step()

        accu = train_protopnet(
            args=args,
            epoch=epoch,
            ppnet_multi=ppnet_multi,
            optimizer=joint_optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"{epoch}",
            class_specific=class_specific,
            log=log,
            tensorboard_writer=tensorboard_writer,
        )
        epoch += 1

    save.save_model_w_condition(
        model=ppnet_multi.module,
        model_dir=log.checkpoint_dir,
        model_name="final",
        accu=accu,
        target_accu=0.0,
        log=log,
    )


def _train_explainable_model(log, tensorboard_writer, args):
    img_dir = log.log_dir / args.dir_for_saving_images

    (
        train_loader,
        train_push_loader,
        test_loader,
        classes,
    ) = get_dataloaders(log, args)

    tensorboard_writer.add_text("Step size", f"Training steps: {len(train_loader)}")
    tensorboard_writer.add_text("Step size", f"Test steps: {len(test_loader)}")

    # we should look into distributed sampler more carefully
    # at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log.info(f"training set size: {len(train_loader.dataset)}")
    log.info(f"push set size: {len(train_push_loader.dataset)}")
    log.info(f"test set size: {len(test_loader.dataset)}")
    log.info(f"batch size: {args.batch_size}")
    log.info(f"number of prototypes per class: {args.n_prototypes_per_class}")

    # construct the model
    ppnet = model.construct_PPNet(
        base_architecture=args.net,
        backbone_only=args.backbone_only,
        pretrained=not args.disable_pretrained,
        img_shape=args.img_shape,
        prototype_shape=args.prototype_shape,
        num_classes=args.num_classes,
        prototype_activation_function=args.prototype_activation_function,
        add_on_layers_type=args.add_on_layers_type,
    )
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet, args.device_ids)
    class_specific = True

    joint_optimizer_specs = [
        {
            "params": ppnet.features.parameters(),
            "lr": args.joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": args.joint_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.prototype_vectors,
            "lr": args.joint_optimizer_lrs["prototype_vectors"],
        },
    ]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=args.joint_lr_step, gamma=0.1
    )

    warm_optimizer_specs = [
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": args.warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.prototype_vectors,
            "lr": args.warm_optimizer_lrs["prototype_vectors"],
        },
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {
            "params": ppnet.last_layer.parameters(),
            "lr": args.finetune_lr,
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    partial_preprocess = partial(
        preprocess, mean=args.mean, std=args.std, in_channels=args.color_channels
    )

    # train the model
    log.info("start training")

    epoch = 1
    accu = 0.0
    while epoch <= args.n_epochs:
        log.info(
            f"{'warm ' if epoch <=args.epochs_warm else ''}"
            f"epoch: \t{epoch} / "
            f"{args.epochs_warm if epoch <= args.epochs_warm else args.n_epochs}"
        )
        if epoch <= args.epochs_warm:
            tnt.warm_only(model=ppnet_multi, log=log)
            optimizer = warm_optimizer
        else:
            tnt.joint(model=ppnet_multi, log=log)
            if epoch > args.epochs_warm:
                joint_lr_scheduler.step()
            optimizer = joint_optimizer

        accu = train_protopnet(
            args=args,
            epoch=epoch,
            ppnet_multi=ppnet_multi,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"{epoch}_no-push_",
            class_specific=class_specific,
            log=log,
            tensorboard_writer=tensorboard_writer,
        )
        epoch += 1

        if epoch in args.push_epochs:
            push.push_prototypes(
                train_push_loader,  # dataloader (must be un-normalized in [0,1])
                prototype_network_parallel=ppnet_multi,
                # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=partial_preprocess,  # normalize
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                # if not None, prototypes will be saved here
                epoch_number=epoch,
                # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=args.prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=args.prototype_self_act_filename_prefix,  # noqa
                proto_bound_boxes_filename_prefix=args.proto_bound_boxes_filename_prefix,  # noqa
                save_prototype_class_identity=True,
                log=log,
                tensorboard_writer=tensorboard_writer,
            )
            accu = train_protopnet(
                args=args,
                epoch=0,
                ppnet_multi=ppnet_multi,
                test_only=True,
                test_loader=test_loader,
                model_name=f"{epoch}_push_",
                class_specific=class_specific,
                log=log,
                tensorboard_writer=None,
            )

            if args.prototype_activation_function != "linear":
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(args.epochs_finetune):
                    log.info(f"finetune iteration: \t{i + 1} / {args.epochs_finetune}")
                    accu = train_protopnet(
                        args=args,
                        epoch=epoch,
                        ppnet_multi=ppnet_multi,
                        optimizer=last_layer_optimizer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        model_name=f"{epoch}_finetune_",
                        class_specific=class_specific,
                        log=log,
                        tensorboard_writer=tensorboard_writer,
                    )
                    epoch += 1

    save.save_model_w_condition(
        model=ppnet_multi.module,
        model_dir=log.checkpoint_dir,
        model_name="final",
        accu=accu,
        target_accu=0.0,
        log=log,
    )

    tensorboard_writer.close()


def train_model(log, args):
    log.info(
        f"Device used: {args.device} "
        f"{f'with id {args.device_ids}' if len(args.device_ids) > 0 else ''}"
    )

    tensorboard_writer = SummaryWriter(log_dir=args.log_dir)

    if args.backbone_only:
        _train_backbone_model(log, tensorboard_writer, args)
    else:
        _train_explainable_model(log, tensorboard_writer, args)

    tensorboard_writer.close()


def train_protopnet(
    args: argparse.Namespace,
    epoch: int,
    ppnet_multi: nn.DataParallel,
    test_loader: torch.utils.data.DataLoader,
    model_name: str,
    class_specific: bool,
    log: Log,
    tensorboard_writer: SummaryWriter,
    test_only: bool = False,
    optimizer: torch.optim.Optimizer = None,
    train_loader: torch.utils.data.DataLoader = None,
):
    if not test_only:
        _ = tnt.train(
            args=args,
            epoch=epoch,
            model=ppnet_multi,
            dataloader=train_loader,
            optimizer=optimizer,
            class_specific=class_specific,
            log=log,
            tensorboard_writer=tensorboard_writer,
        )
    accu = tnt.test(
        args=args,
        epoch=epoch,
        model=ppnet_multi,
        dataloader=test_loader,
        class_specific=class_specific,
        log=log,
        tensorboard_writer=tensorboard_writer,
    )
    save.save_model_w_condition(
        model=ppnet_multi.module,
        model_dir=log.checkpoint_dir,
        model_name=model_name,
        accu=accu,
        target_accu=0.60 if not args.save_all_models else 0.0,
        log=log,
    )

    return accu
