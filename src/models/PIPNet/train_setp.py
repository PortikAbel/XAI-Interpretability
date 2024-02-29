from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data


def train_pipnet(
    net,
    train_loader,
    optimizer_net,
    optimizer_classifier,
    scheduler_net,
    scheduler_classifier,
    criterion,
    epoch,
    args,
    device,
    log,
    tensorboard_writer=None,
    pretrain=False,
    finetune=False,
    progress_prefix: str = "Train Epoch",
):
    # Make sure the model is in train mode
    net.train()

    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = "Pretrain Epoch"
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

    iters = len(train_loader)
    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=iters,
        desc=progress_prefix + "%s" % epoch,
        mininterval=2.0,
        ncols=0,
        file=log.tqdm_file,
    )

    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch / args.epochs_pretrain) * 1.0
        t_weight = 5.0
        unif_weight = 0.5
        var_weigth = 0.5
        cl_weight = 0.0
    else:
        align_pf_weight = 5.0
        t_weight = 2.0
        unif_weight = 2.0
        var_weigth = 2.0
        cl_weight = 2.0
    if not args.tanh_loss:
        t_weight = 0.0
    if not args.unif_loss:
        unif_weight = 0.0
    if not args.variance_loss:
        var_weigth = 0.0

    print(
        f"Align weight: {align_pf_weight}, "
        f"U_tanh weight: {t_weight}, "
        f"Uniformity weight: {unif_weight}, "
        f"Variance weight: {var_weigth}",
        f"Class weight: {cl_weight}",
        flush=True,
    )
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)

    lrs_net = []
    lrs_class = []
    prototype_activations = torch.empty(
        (iters, 2 * train_loader.batch_size, net.module._num_prototypes)
    )
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        prototype_activations[i, :, :] = pooled

        loss, acc = calculate_loss(
            proto_features,
            pooled,
            out,
            ys,
            align_pf_weight,
            t_weight,
            unif_weight,
            var_weigth,
            cl_weight,
            net.module._classification.normalization_multiplier,
            pretrain,
            finetune,
            criterion,
            train_iter,
            tensorboard_writer,
            len(train_iter) * (epoch - 1) + i,
            print=True,
            EPS=1e-8,
        )

        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.0)

        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(
                    torch.where(
                        net.module._classification.weight < 1e-3,
                        0.0,
                        net.module._classification.weight,
                    )
                )  # set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(
                    torch.clamp(
                        net.module._classification.normalization_multiplier.data,
                        min=1.0,
                    )
                )
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(
                        torch.clamp(net.module._classification.bias.data, min=0.0)
                    )
    train_info["train_accuracy"] = total_acc / float(i + 1)
    train_info["loss"] = total_loss / float(i + 1)
    train_info["lrs_net"] = lrs_net
    train_info["lrs_class"] = lrs_class
    train_info["prototype_activations"] = (
        prototype_activations.view((-1, net.module._num_prototypes))
        .detach()
        .cpu()
        .numpy()
    )

    return train_info


def calculate_loss(
    proto_features,
    pooled,
    out,
    ys1,
    align_pf_weight,
    t_weight,
    unif_weight,
    var_weigth,
    cl_weight,
    net_normalization_multiplier,
    pretrain,
    finetune,
    criterion,
    train_iter,
    tensorboard_writer=None,
    iteration=0,
    print=True,
    EPS=1e-10,
):
    ys = torch.cat([ys1, ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (
        align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())
    ) / 2.0
    tanh_loss = (
        -(
            torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean()
            + torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()
        )
        / 2.0
    )
    uni_loss = (
        uniform_loss(F.normalize(pooled1 + EPS, dim=1))
        + uniform_loss(F.normalize(pooled2 + EPS, dim=1))
    ) / 2.0
    var_loss = (variance_loss(embv1) + variance_loss(embv2)) / 2.0

    if not finetune:
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss
        loss += unif_weight * uni_loss
        loss += var_weigth * var_loss

    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys)

        if finetune:
            loss = cl_weight * class_loss
        else:
            loss += cl_weight * class_loss

    acc = 0.0
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    if print:
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                    (
                        f"L: {loss.item():.3f}, "
                        f"LA:{a_loss_pf.item():.2f}, "
                        f"LT:{tanh_loss.item():.3f}, "
                        f"LU:{uni_loss.item():.3f}, "
                        f"LV:{var_loss.item():.3f}, "
                        f"num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}"  # noqa
                    ),
                    refresh=False,
                )
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/pretrain/L", loss.item(), iteration)
                    tensorboard_writer.add_scalar("Loss/pretrain/LA", a_loss_pf.item(), iteration)
                    if t_weight > 0:
                        tensorboard_writer.add_scalar("Loss/pretrain/LT", tanh_loss.item(), iteration)
                    if unif_weight > 0:
                        tensorboard_writer.add_scalar("Loss/pretrain/LU", uni_loss.item(), iteration)
                    if var_weigth > 0:
                        tensorboard_writer.add_scalar("Loss/pretrain/LV", var_loss.item(), iteration)
            else:
                train_iter.set_postfix_str(
                    (
                        f"L:{loss.item():.3f}, "
                        f"LC:{class_loss.item():.3f}, "
                        f"LA:{a_loss_pf.item():.2f}, "
                        f"LT:{tanh_loss.item():.3f}, "
                        f"LU:{uni_loss.item():.3f}, "
                        f"LV:{var_loss.item():.3f}, "
                        f"num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, "  # noqa
                        f"Ac:{acc:.3f}"
                    ),
                    refresh=False,
                )
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/train/L", loss.item(), iteration)
                    tensorboard_writer.add_scalar("Loss/train/LA", a_loss_pf.item(), iteration)
                    tensorboard_writer.add_scalar("Loss/train/LC", class_loss.item(), iteration)
                    if t_weight > 0:
                        tensorboard_writer.add_scalar("Loss/train/LT", tanh_loss.item(), iteration)
                    if unif_weight > 0:
                        tensorboard_writer.add_scalar("Loss/train/LU", uni_loss.item(), iteration)
                    if var_weigth > 0:
                        tensorboard_writer.add_scalar("Loss/train/LV", var_loss.item(), iteration)
                    tensorboard_writer.add_scalar("Acc/train", acc, iteration)

    return loss, acc


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/.
# Currently not used but you could try adding it if you want.
def uniform_loss(x, t=2, EPS=1e-10):
    # print(
    #   "sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape,
    #   torch.sum(torch.pow(x,2), dim=1),
    # ) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + EPS).log()
    return loss


def variance_loss(x, gamma=1, EPS=1e-12):
    return (gamma - (x.var(dim=0) + EPS).sqrt()).clamp(min=0).mean()


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert not targets.requires_grad

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss
