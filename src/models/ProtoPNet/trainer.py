from functools import partial

import torch

from models.ProtoPNet import model, push
from models.ProtoPNet import train_and_test as tnt
from models.ProtoPNet.util import save
from models.ProtoPNet.util.data import get_dataloaders
from models.ProtoPNet.util.preprocess import preprocess


def train_model(log, args):
    print(
        f"Device used: {args.device} "
        f"{f'with id {args.device_ids}' if len(args.device_ids) > 0 else ''}",
        flush=True,
    )

    img_dir = log.log_dir / args.dir_for_saving_images

    (
        train_loader,
        train_push_loader,
        test_loader,
        classes,
    ) = get_dataloaders(args)

    # we should look into distributed sampler more carefully
    # at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log(f"training set size: {len(train_loader.dataset)}")
    log(f"push set size: {len(train_push_loader.dataset)}")
    log(f"test set size: {len(test_loader.dataset)}")
    log(f"batch size: {args.batch_size}")
    log(f"number of prototypes per class: {args.n_prototypes_per_class}")

    # construct the model
    ppnet = model.construct_PPNet(
        base_architecture=args.net,
        pretrained=not args.disable_pretrained,
        img_size=256,
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
    joint_optimizer_specs += [
        {
            "params": ppnet.prototype_vectors,
            "lr": args.joint_optimizer_lrs["prototype_vectors"],
        },
    ]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=args.joint_lr_step_size, gamma=0.1
    )

    warm_optimizer_specs = [
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": args.warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]
    warm_optimizer_specs += [
        {
            "params": ppnet.prototype_vectors,
            "lr": args.warm_optimizer_lrs["prototype_vectors"],
        },
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {
            "params": ppnet.last_layer.parameters(),
            "lr": args.last_layer_optimizer_lr,
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    partial_preprocess = partial(
        preprocess, mean=args.mean, std=args.std, in_channels=args.color_channels
    )

    # train the model
    log("start training")

    for epoch in range(args.n_epochs):
        log(f"epoch: \t{epoch}")

        if epoch < args.epochs_warm:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(
                args=args,
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                class_specific=class_specific,
                log=log,
            )
        else:
            tnt.joint(model=ppnet_multi, log=log)
            if epoch > 0:
                joint_lr_scheduler.step()
            _ = tnt.train(
                args=args,
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=joint_optimizer,
                class_specific=class_specific,
                log=log,
            )

        accu = tnt.test(
            args=args,
            model=ppnet_multi,
            dataloader=test_loader,
            class_specific=class_specific,
            log=log,
        )
        save.save_model_w_condition(
            model=ppnet,
            model_dir=log.checkpoint_dir,
            model_name=f"{epoch}_no-push_",
            accu=accu,
            target_accu=0.60,
            log=log,
        )

        if epoch >= args.push_start and epoch in args.push_epochs:
            push.push_prototypes(
                train_push_loader,  # pytorch dataloader (must be un-normalized in [0,1])
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
                prototype_self_act_filename_prefix=args.prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=args.proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
            )
            accu = tnt.test(
                args=args,
                model=ppnet_multi,
                dataloader=test_loader,
                class_specific=class_specific,
                log=log,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=log.checkpoint_dir,
                model_name=f"{epoch}_push_",
                accu=accu,
                target_accu=0.60,
                log=log,
            )

            if args.prototype_activation_function != "linear":
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(args.epochs_finetune):
                    log(f"iteration: \t{i}")
                    _ = tnt.train(
                        args=args,
                        model=ppnet_multi,
                        dataloader=train_loader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        log=log,
                    )
                    accu = tnt.test(
                        args=args,
                        model=ppnet_multi,
                        dataloader=test_loader,
                        class_specific=class_specific,
                        log=log,
                    )
                    save.save_model_w_condition(
                        model=ppnet,
                        model_dir=log.checkpoint_dir,
                        model_name=f"{epoch}_push_{i}_",
                        accu=accu,
                        target_accu=0.60,
                        log=log,
                    )
