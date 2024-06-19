import argparse
from collections import defaultdict
import random
import numpy as np
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw as D
from tqdm import tqdm

from models.PIPNet.util.log import Log
from models.PIPNet.visualize import get_patch_size, get_patch


@torch.no_grad()
def visualize_top_k(
    net,
    project_loader,
    num_classes,
    device,
    folder_name,
    args: argparse.Namespace,
    log: Log,
    k=10,
):
    print("Visualizing prototypes for top k...", flush=True)
    result_dir = args.log_dir / folder_name
    result_dir.mkdir(parents=True, exist_ok=True)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()

    for p in range(net.module._num_prototypes):
        near_imgs_dir = result_dir / str(p)
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.0
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []

    patch_size, skip = get_patch_size(args)

    imgs = project_loader.dataset.imgs

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = tqdm(
        enumerate(project_loader),
        total=len(project_loader),
        mininterval=50.0,
        desc="Collecting topk",
        ncols=0,
        file=log.tqdm_file,
    )

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, ys) in img_iter:
        images_seen += 1
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            pfs, pooled, _ = net(xs, inference=True)
            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)

            for p in range(pooled.shape[0]):
                c_weight = torch.max(classification_weights[:, p])
                if c_weight > 1e-3:
                    # ignore prototypes that are not relevant to any class
                    if p not in topks.keys():
                        topks[p] = []

                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(
                            topks[p], key=lambda tup: tup[1], reverse=True
                        )
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one
                            # (since dataset is not shuffled so latter images
                            # with same scores can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())

    prototypes_not_used = []
    i_to_p: dict = defaultdict(list)
    for p in topks.keys():
        img_idxs, scores = zip(*topks[p])
        if any(np.array(scores) > 0.1):
            for i in img_idxs:
                i_to_p[i].append(p)
        else:
            prototypes_not_used.append(p)

    print(
        len(prototypes_not_used),
        "prototypes do not have any similarity score > 0.1. "
        "Will be ignored in visualisation.",
    )
    abstained = 0
    # Show progress on progress bar
    img_iter = tqdm(
        enumerate(project_loader),
        total=len(project_loader),
        mininterval=50.0,
        desc="Visualizing topk",
        ncols=0,
        file=log.tqdm_file,
    )
    for i, (xs, ys) in img_iter:
        # shuffle is false so should lead to same order as in imgs
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, pooled, out = net(
                xs, inference=True
            )  # softmaxes has shape (1, num_prototypes, W, H)

            # shape ([1]) because batch size of projectloader is 1
            outmax = torch.amax(out, dim=1)[0]
            if outmax.item() == 0.0:
                abstained += 1

        for p in i_to_p[i]:
            c_weight = torch.max(
                classification_weights[:, p]
            )  # ignore prototypes that are not relevant to any class
            if (c_weight > 1e-10) or ("pretrain" in folder_name):
                # get the h and w index of the max prototype from the p slice
                proto_slice = softmaxes[0, p, :, :]
                h_idx, w_idx = (proto_slice.max() == proto_slice).nonzero(as_tuple=True)
                h_idx, w_idx = h_idx[0], w_idx[0]
                img_to_open = imgs[i]
                if isinstance(img_to_open, tuple) or isinstance(
                    img_to_open, list
                ):  # dataset contains tuples of (img,label)
                    img_to_open = img_to_open[0]

                image, img_tensor_patch, _, _, _, _ = get_patch(
                    img_to_open, args, h_idx, w_idx, softmaxes.shape
                )

                saved[p] += 1
                tensors_per_prototype[p].append(img_tensor_patch)

    print("Abstained: ", abstained, flush=True)
    all_tensors = []
    for p in range(net.module._num_prototypes):
        if saved[p] > 0:
            # add text next to each topk-grid, to easily see which prototype it is
            text = "P " + str(p)
            txt_image = Image.new(
                "RGB",
                (img_tensor_patch.shape[1], img_tensor_patch.shape[2]),
                (0, 0, 0),
            )
            draw = D.Draw(txt_image)
            draw.text(
                (
                    img_tensor_patch.shape[0] // 2,
                    img_tensor_patch.shape[1] // 2,
                ),
                text,
                anchor="mm",
                fill="white",
            )
            txt_tensor = transforms.ToTensor()(txt_image)
            tensors_per_prototype[p].append(txt_tensor)
            # save top-k image patches in grid
            try:
                grid = torchvision.utils.make_grid(
                    tensors_per_prototype[p], nrow=k + 1, padding=1
                )
                torchvision.utils.save_image(grid, result_dir / f"grid_top_k_{p}.png")
                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]
            except Exception:
                pass
    if len(all_tensors) > 0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
        torchvision.utils.save_image(grid, result_dir / "grid_top_k_all.png")
    else:
        print(
            "Pretrained prototypes not visualized. Try to pretrain longer.",
            flush=True,
        )
    return topks


def visualize(
    net,
    projectloader,
    num_classes,
    device,
    foldername,
    args: argparse.Namespace,
    log: Log,
):
    print("Visualizing prototypes...", flush=True)
    result_dir = args.log_dir / foldername
    result_dir.mkdir(parents=True, exist_ok=True)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()

    for p in range(net.module._num_prototypes):
        near_imgs_dir = result_dir / str(p)
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.0
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []

    imgs = projectloader.dataset.imgs

    # skip some images for visualisation to speed up the process
    if len(imgs) / num_classes < 10:
        skip_img = 10
    elif len(imgs) / num_classes < 50:
        skip_img = 5
    else:
        skip_img = 2
    print(
        "Every",
        skip_img,
        "is skipped in order to speed up the visualisation process",
        flush=True,
    )

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=100.0,
        desc="Visualizing",
        ncols=0,
        file=log.tqdm_file,
    )

    # Iterate through the data
    images_seen_before = 0
    for i, (
        xs,
        ys,
    ) in img_iter:  # shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before += xs.shape[0]
            continue

        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True)

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(
            max_per_prototype, dim=1
        )
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(
            max_per_prototype_h, dim=1
        )
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(
                classification_weights[:, p]
            )  # ignore prototypes that are not relevant to any class
            if c_weight > 0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p, h_idx, w_idx].item()
                found_max = max_per_prototype[p, h_idx, w_idx].item()

                img_name = imgs[images_seen_before + idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(img_name)
                else:
                    notabstainedimgs.add(img_name)

                if found_max > seen_max[p]:
                    seen_max[p] = found_max

                if found_max > 0.5:
                    img_to_open, img_label = imgs[images_seen_before + idx_to_select]
                    img_to_open = Path(img_to_open).resolve()
                    (
                        image,
                        img_tensor_patch,
                        h_coord_max,
                        h_coord_min,
                        w_coord_max,
                        w_coord_min,
                    ) = get_patch(img_to_open, args, h_idx, w_idx, softmaxes.shape)
                    saved[p] += 1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))

                    save_path = result_dir / f"prototype_{p}"
                    save_path.mkdir(parents=True, exist_ok=True)

                    draw = D.Draw(image)
                    draw.rectangle(
                        ((w_coord_min, h_coord_min), (w_coord_max, h_coord_max)),
                        outline="yellow",
                        width=2,
                    )
                    image.save(
                        save_path
                        / f"p{p}_{img_label}_{round(found_max, 2)}_{img_to_open.stem}"
                        f"_rect.png"
                    )

        images_seen_before += len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p] > 0:
            try:
                sorted_by_second = sorted(
                    tensors_per_prototype[p],
                    key=lambda tup: tup[1],
                    reverse=True,
                )
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid, result_dir / f"grid_{p}.png")
            except RuntimeError:
                pass
