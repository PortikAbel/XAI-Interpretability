from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import models.PIPNet.visualize.logs as visual_logs
from models.PIPNet.pipnet import PIPNet, get_network
from models.PIPNet.test_setp import eval_ood, eval_pipnet, get_thresholds
from models.PIPNet.train_setp import train_pipnet
from models.PIPNet.util.args import PIPNetArgumentParser
from models.PIPNet.util.data import get_dataloaders
from models.PIPNet.util.eval_cub_csv import (
    eval_prototypes_cub_parts_csv,
    get_proto_patches_cub,
    get_topk_cub,
)
from models.PIPNet.util.func import init_weights_xavier
from models.PIPNet.visualize.pipnet import visualize, visualize_top_k
from models.PIPNet.visualize.prediction import vis_pred, vis_pred_experiments
from utils.environment import get_env


def train_model(log, args=None):
    args = args or PIPNetArgumentParser.get_args()

    tensorboard_writer = SummaryWriter(log_dir=args.log_dir)

    # Log which device was actually used
    log.info(
        f"Device used: {args.device} "
        f"{f'with id {args.device_ids}' if len(args.device_ids) > 0 else ''}",
    )

    # Obtain the dataloaders
    (
        train_loader,
        train_loader_pretraining,
        project_loader,
        test_loader,
        test_project_loader,
        classes,
    ) = get_dataloaders(log, args)

    if len(classes) <= 20:
        if args.validation_size == 0.0:
            # print("Classes: ", test_loader.dataset.class_to_idx, flush=True)
            pass
        else:
            print("Classes: ", str(classes), flush=True)

    # Create a convolutional network based on arguments and add 1x1 conv layer
    (
        feature_net,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes,
    ) = get_network(len(classes), args)

    # Create a PIP-Net
    net = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )

    net = net.to(device=args.device)
    net = nn.DataParallel(net, device_ids=args.device_ids)

    optimizer_net, optimizer_classifier = net.module.get_optimizers()

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net is not None:
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net, map_location=args.device)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])
            except Exception:
                pass
            if (
                torch.mean(net.module._classification.weight).item() > 1.0
                and torch.mean(net.module._classification.weight).item() < 3.0
                and torch.count_nonzero(
                    torch.relu(net.module._classification.weight - 1e-5)
                )
                .float()
                .item()
                > 0.8 * (num_prototypes * len(classes))
            ):  # assume that the linear classification layer is not yet trained
                # (e.g. when loading a pretrained backbone only)
                print(
                    "We assume that the classification layer is not yet trained. "
                    "We re-initialize it...",
                    flush=True,
                )
                torch.nn.init.normal_(
                    net.module._classification.weight, mean=1.0, std=0.1
                )
                torch.nn.init.constant_(net.module._multiplier, val=2.0)
                print(
                    "Classification layer initialized with mean",
                    torch.mean(net.module._classification.weight).item(),
                    flush=True,
                )
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            else:
                if "optimizer_classifier_state_dict" in checkpoint.keys():
                    optimizer_classifier.load_state_dict(
                        checkpoint["optimizer_classifier_state_dict"]
                    )

        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            torch.nn.init.constant_(net.module._multiplier, val=2.0)
            net.module._multiplier.requires_grad = False

            print(
                "Classification layer initialized with mean",
                torch.mean(net.module._classification.weight).item(),
                flush=True,
            )

    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction="mean").to(args.device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(train_loader_pretraining) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(train_loader))
        xs1 = xs1.to(args.device)
        proto_features, _, _ = net(xs1)
        wshape = np.array(proto_features.shape)[-2:]
        args.wshape = wshape  # needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)

    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy,
        # F1-score, mean train accuracy and mean loss for each epoch
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_f1",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )
        print(
            "Your dataset only has two classes. "
            "Is the number of samples per class similar? "
            "If the data is imbalanced, we recommend to use "
            "the --weighted_loss flag to account for the imbalance.",
            flush=True,
        )
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5),
        # mean train accuracy and mean loss for each epoch
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_top5_acc",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )

    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain + 1):
        print(
            "\nPretrain Epoch",
            epoch,
            "with batch size",
            train_loader_pretraining.batch_size,
            flush=True,
        )

        # Pretrain prototypes
        net.module.pretrain()
        train_info = train_pipnet(
            net,
            train_loader_pretraining,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            None,
            criterion,
            epoch,
            args,
            args.device,
            log,
            tensorboard_writer,
            pretrain=True,
            finetune=False,
        )
        visual_logs.class_weights_heatmap(
            tensorboard_writer=tensorboard_writer,
            net=net,
            epoch=epoch,
        )
        if args.log_prototype_activations_violin_plot:
            visual_logs.prototype_activations_violin_plot(
                tensorboard_writer=tensorboard_writer,
                net=net,
                epoch=epoch,
                train_info=train_info,
            )
        lrs_pretrain_net += train_info["lrs_net"]
        visual_logs.save_lr_curve(
            learning_rates=lrs_pretrain_net,
            save_path=args.log_dir / "lr_pretrain_net.png",
        )
        log.log_values(
            "log_epoch_overview",
            epoch,
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            train_info["loss"],
        )

    def get_checkpoint(with_optimizer_classifier_state_dict: bool = True):
        if not with_optimizer_classifier_state_dict:
            return {
                "model_state_dict": net.state_dict(),
                "optimizer_net_state_dict": optimizer_net.state_dict(),
            }
        return {
            "model_state_dict": net.state_dict(),
            "optimizer_net_state_dict": optimizer_net.state_dict(),
            "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
        }

    if args.state_dict_dir_net is None:
        net.eval()
        torch.save(
            get_checkpoint(with_optimizer_classifier_state_dict=False),
            args.log_dir / "checkpoints" / "net_pretrained",
        )
        net.train()
    with torch.no_grad():
        if args.visualize_topk and "convnext" in args.net and args.epochs_pretrain > 0:
            topks = visualize_top_k(
                net,
                project_loader,
                len(classes),
                args.device,
                "visualised_pretrained_prototypes_topk",
                args,
                log,
            )

    # SECOND TRAINING PHASE re-initialize optimizers and schedulers
    # for second training phase
    optimizer_net, optimizer_classifier = net.module.get_optimizers()
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(train_loader) * args.epochs,
        eta_min=args.lr_net / 100.0,
    )
    # scheduler for the classification layer is with restarts,
    # such that the model can re-activated zeroed-out prototypes.
    # Hence, an intuitive choice.
    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier,
            T_0=5,
            eta_min=0.001,
            T_mult=1,
            verbose=False,
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier,
            T_0=10,
            eta_min=0.001,
            T_mult=1,
            verbose=False,
        )

    frozen = True
    lrs_net = []
    lrs_classifier = []

    for epoch in range(1, args.epochs + 1):
        # during fine-tuning, only train classification layer and freeze rest.
        # usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= args.epochs_finetune and (
            args.epochs_pretrain > 0 or args.state_dict_dir_net is not None
        ):
            net.module.finetune()
            finetune = True

        # freeze first layers of backbone, train rest
        elif epoch <= args.freeze_epochs:
            finetune = False
            net.module.freeze()
            frozen = True

        # unfreeze backbone
        else:
            net.module.unfreeze()
            frozen = False

        print("\n Epoch", epoch, "frozen:", frozen, flush=True)
        visual_logs.class_weights_heatmap(
            tensorboard_writer=tensorboard_writer,
            net=net,
            epoch=epoch,
        )
        if (epoch == args.epochs or epoch % 30 == 0) and args.epochs > 1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 0.001, min=0.0)
                )
                cls_w = net.module._classification.weight[
                    net.module._classification.weight.nonzero(as_tuple=True)
                ]
                print(f"Classifier weights: {cls_w}\n{cls_w.shape}", flush=True)
                if args.bias:
                    cls_b = net.module._classification.bias
                    print(f"Classifier bias: {cls_b}", flush=True)
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(
            net,
            train_loader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            args,
            args.device,
            log,
            tensorboard_writer,
            pretrain=False,
            finetune=finetune,
        )
        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]
        # Evaluate model
        eval_info = eval_pipnet(net, test_loader, epoch, args.device, log)
        log.log_values(
            "log_epoch_overview",
            epoch,
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            train_info["train_accuracy"],
            train_info["loss"],
        )
        tensorboard_writer.add_scalar(
            "Acc/eval-epochs", eval_info["top1_accuracy"], epoch
        )
        tensorboard_writer.add_scalar(
            "Acc/train-epochs", train_info["train_accuracy"], epoch
        )
        tensorboard_writer.add_scalar("Loss/train-epochs", train_info["loss"], epoch)
        tensorboard_writer.add_scalar(
            "Num non-zero prototypes", eval_info["almost_nonzeros"], epoch
        )

        with torch.no_grad():
            net.eval()
            torch.save(
                get_checkpoint(),
                args.log_dir / "checkpoints" / "net_trained",
            )

            if epoch % 30 == 0:
                net.eval()
                torch.save(
                    get_checkpoint(),
                    args.log_dir / "checkpoints" / f"net_trained_{epoch}",
                )

            visual_logs.save_lr_curve(
                learning_rates=lrs_net,
                save_path=args.log_dir / "lr_net.png",
            )
            visual_logs.save_lr_curve(
                learning_rates=lrs_classifier,
                save_path=args.log_dir / "lr_class.png",
            )

    net.eval()
    torch.save(
        get_checkpoint(),
        args.log_dir / "checkpoints" / "net_trained_last",
    )
    topks = None
    if args.visualize_topk:
        topks = visualize_top_k(
            net,
            project_loader,
            len(classes),
            args.device,
            "visualised_prototypes_topk",
            args,
            log,
        )
        # set weights of prototypes that are never really found in projection set to 0
        set_to_zero = []
        for prot in topks.keys():
            found = False
            for i_id, score in topks[prot]:
                if score > 0.1:
                    found = True
            if not found:
                torch.nn.init.zeros_(net.module._classification.weight[:, prot])
                set_to_zero.append(prot)
        print(
            f"Weights of prototypes {set_to_zero} are set to zero because"
            " it is never detected with similarity>0.1 in the training set",
            flush=True,
        )
        eval_info = eval_pipnet(
            net, test_loader, "notused" + str(args.epochs), args.device, log
        )
        log.log_values(
            "log_epoch_overview",
            "notused" + str(args.epochs),
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            "n.a.",
            "n.a.",
        )

    print("classifier weights: ", net.module._classification.weight, flush=True)
    print(
        "Classifier weights nonzero: ",
        net.module._classification.weight[
            net.module._classification.weight.nonzero(as_tuple=True)
        ],
        (
            net.module._classification.weight[
                net.module._classification.weight.nonzero(as_tuple=True)
            ]
        ).shape,
        flush=True,
    )
    print("Classifier bias: ", net.module._classification.bias, flush=True)
    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c, :]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        # if args.validation_size == 0.0:
        #     print(
        #         "Class",
        #         c,
        #         "(",
        #         list(test_loader.dataset.class_to_idx.keys())[
        #             list(test_loader.dataset.class_to_idx.values()).index(c)
        #         ],
        #         "):",
        #         "has",
        #         len(relevant_ps),
        #         "relevant prototypes: ",
        #         relevant_ps,
        #         flush=True,
        #     )

    # Evaluate prototype purity
    if args.evaluate_purity and args.dataset.startswith("CUB"):
        project_path = Path(get_env("DATA_ROOT"), "CUB_200")
        parts_loc_path = project_path / "parts/part_locs.txt"
        parts_name_path = project_path / "parts/parts.txt"
        imgs_id_path = project_path / "images.txt"
        cubthreshold = 0.5

        net.eval()
        print("\n\nEvaluating cub prototypes for training set", flush=True)
        csvfile_topk = get_topk_cub(
            net, project_loader, 10, f"train_{epoch}", args.device, args, log
        )
        eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            f"train_topk_{epoch}",
            args,
            log,
        )

        csvfile_all = get_proto_patches_cub(
            net,
            project_loader,
            f"train_all_{epoch}",
            args.device,
            args,
            log,
            threshold=cubthreshold,
        )
        eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            f"train_all_thres{cubthreshold}_{epoch}",
            args,
            log,
        )

        print("\n\nEvaluating cub prototypes for test set", flush=True)
        csvfile_topk = get_topk_cub(
            net, test_project_loader, 10, f"test_{epoch}", args.device, args, log
        )
        eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            f"test_topk_{epoch}",
            args,
            log,
        )
        cubthreshold = 0.5
        csvfile_all = get_proto_patches_cub(
            net,
            test_project_loader,
            f"test_{epoch}",
            args.device,
            args,
            log,
            threshold=cubthreshold,
        )
        eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            f"test_all_thres{cubthreshold}_{epoch}",
            args,
            log,
        )

    # visualize predictions
    if args.visualize_predictions:
        visualize(
            net,
            project_loader,
            len(classes),
            args.device,
            "visualised_prototypes",
            args,
            log,
        )
        test_path = Path(test_project_loader.dataset.samples[0][0]).parent.parent
        vis_pred(net, test_path, classes, args.device, args)
        if args.extra_test_image_folder != "":
            if Path(args.extra_test_image_folder).exists():
                vis_pred_experiments(
                    net, args.extra_test_image_folder, classes, args.device, args
                )

    # EVALUATE OOD DETECTION
    if args.evaluate_ood:
        ood_datasets = ["CARS", "CUB-200-2011", "pets"]
        for percent in [95.0]:
            print(
                "\nOOD Evaluation for epoch",
                epoch,
                "with percent of",
                percent,
                flush=True,
            )
            _, _, _, class_thresholds = get_thresholds(
                net, test_loader, epoch, args.device, log, percent
            )
            print("Thresholds:", class_thresholds, flush=True)
            # Evaluate with in-distribution data
            id_fraction = eval_ood(
                net, test_loader, epoch, args.device, class_thresholds, log
            )
            print(
                f"ID class threshold ID fraction (TPR) with "
                f"percent {percent:.2%}: {id_fraction:.4}",
                flush=True,
            )

            # Evaluate with out-of-distribution data
            for ood_dataset in ood_datasets:
                if ood_dataset != args.dataset:
                    print("\n OOD dataset: ", ood_dataset, flush=True)
                    ood_args = deepcopy(args)
                    ood_args.dataset = ood_dataset
                    _, _, _, ood_test_loader, _, _ = get_dataloaders(ood_args)

                    id_fraction = eval_ood(
                        net, ood_test_loader, epoch, args.device, class_thresholds, log
                    )
                    print(
                        f"{args.dataset} - OOD {ood_dataset} class threshold ID "
                        f"fraction (FPR) with percent {percent:.2%}: {id_fraction:.4f}",
                        flush=True,
                    )

    print("Done!", flush=True)
    tensorboard_writer.close()
