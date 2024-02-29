import os
import shutil
import argparse

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw as D
import matplotlib.pyplot as plt

from models.PIPNet.visualize import get_patch_size, get_img_coordinates

try:
    import cv2

    use_opencv = True
except ImportError:
    use_opencv = False
    print(
        "Heatmaps showing where a prototype is found will not be generated "
        "because OpenCV is not installed.",
        flush=True,
    )


def __visualize_predictions(
    net,
    image_dir,
    classes,
    device,
    args: argparse.Namespace,
    out_dir=None,
    save_heatmaps=False,
    images_per_class=None,
    top_k=None,
):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = args.log_dir / args.dir_for_saving_images
    if out_dir is not None:
        match out_dir:  # noqa
            case str():
                save_dir = save_dir / out_dir
            case list():
                for o_dir in out_dir:
                    save_dir = save_dir / o_dir
            case _:
                raise ValueError("out_dir must be a string or a list of strings")

    if save_dir.exists():
        shutil.rmtree(save_dir)

    patch_size, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose(
        [
            transforms.Resize(size=tuple(args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    vis_test_set = torchvision.datasets.ImageFolder(
        image_dir, transform=transform_no_augment
    )
    vis_test_loader = DataLoader(
        vis_test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=not args.disable_cuda and torch.cuda.is_available(),
        num_workers=num_workers,
    )
    imgs = vis_test_set.imgs

    last_y = -1
    count_per_y = 0
    for k, (xs, ys) in enumerate(
        vis_test_loader
    ):  # shuffle is false so should lead to same order as in imgs
        if ys[0] != last_y:
            last_y = ys[0]
            count_per_y = 0
        else:
            count_per_y += 1
            if images_per_class is not None and count_per_y > images_per_class:
                # show max images_per_class (5) images per class to speed up the process
                continue
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        new_location = save_dir / img_name
        if not new_location.exists():
            new_location.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, new_location)

        with torch.no_grad():
            softmaxes, pooled, out = net(
                xs, inference=True
            )  # softmaxes has shape (bs, num_prototypes, W, H),
            # pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            for pred_class_idx in sorted_out_indices[:top_k]:
                # process the first top_k (3) element
                # if None the whole list is processed
                pred_class = classes[pred_class_idx]
                save_path = (
                    new_location / f"{pred_class}_{out[0, pred_class_idx].item():.3f}"
                )

                if not save_path.exists():
                    os.makedirs(save_path)

                sorted_pooled, sorted_pooled_indices = torch.sort(
                    pooled.squeeze(0), descending=True
                )
                simweights = []
                for prototype_idx in sorted_pooled_indices:
                    simweight = (
                        pooled[0, prototype_idx].item()
                        * net.module._classification.weight[
                            pred_class_idx, prototype_idx
                        ].item()
                    )
                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        max_h, max_idx_h = torch.max(
                            softmaxes[0, prototype_idx, :, :], dim=0
                        )
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()
                        image = transforms.Resize(size=tuple(args.image_size))(
                            Image.open(img).convert("RGB")
                        )
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(
                            0
                        )  # shape (1, 3, h, w)
                        (
                            h_coor_min,
                            h_coor_max,
                            w_coor_min,
                            w_coor_max,
                        ) = get_img_coordinates(
                            args.image_size,
                            softmaxes.shape,
                            patch_size,
                            skip,
                            max_idx_h,
                            max_idx_w,
                        )
                        img_tensor_patch = img_tensor[
                            0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max
                        ]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)

                        prototype_idx_str = f"p{prototype_idx.item()}"
                        similarity_str = f"sim{pooled[0, prototype_idx].item():.3f}"
                        weight_str = f"w{net.module._classification.weight[pred_class_idx, prototype_idx].item():.3f}"  # noqa
                        file_base = (
                            f"mul{simweight:.3f}_{prototype_idx_str}_"
                            f"{similarity_str}_{weight_str}"
                        )

                        img_patch.save(save_path / f"{file_base}_patch.png")
                        draw = D.Draw(image)
                        draw.rectangle(
                            (
                                (max_idx_w * skip[1], max_idx_h * skip[0]),
                                (
                                    min(
                                        args.image_size[1],
                                        max_idx_w * skip[1] + patch_size,
                                    ),
                                    min(
                                        args.image_size[0],
                                        max_idx_h * skip[1] + patch_size,
                                    ),
                                ),
                            ),
                            outline="yellow",
                            width=2,
                        )
                        image.save(save_path / f"{file_base}_rect.png")

                        # visualise softmaxes as heatmap
                        if use_opencv and save_heatmaps:
                            softmaxes_resized = transforms.ToPILImage()(
                                softmaxes[0, prototype_idx, :, :]
                            )
                            softmaxes_resized = softmaxes_resized.resize(
                                tuple(args.image_size),
                                Image.BICUBIC,
                            )
                            softmaxes_np = (
                                (transforms.ToTensor()(softmaxes_resized))
                                .squeeze()
                                .numpy()
                            )

                            save_heatmaps = cv2.applyColorMap(
                                np.uint8(255 * softmaxes_np), cv2.COLORMAP_JET
                            )
                            save_heatmaps = np.float32(save_heatmaps) / 255
                            save_heatmaps = save_heatmaps[
                                ..., ::-1
                            ]  # OpenCV's BGR to RGB
                            heatmap_img = 0.2 * np.float32(
                                save_heatmaps
                            ) + 0.6 * np.float32(
                                img_tensor.squeeze().numpy().transpose(1, 2, 0)
                            )
                            plt.imsave(
                                fname=save_path / f"heatmap_{prototype_idx_str}.png",
                                arr=heatmap_img,
                                vmin=0.0,
                                vmax=1.0,
                            )


def vis_pred(net, vis_test_dir, classes, device, args: argparse.Namespace):
    __visualize_predictions(
        net,
        vis_test_dir,
        classes,
        device,
        args,
        save_heatmaps=True,
        images_per_class=5,
        top_k=3,
    )


def vis_pred_experiments(net, imgs_dir, classes, device, args: argparse.Namespace):
    __visualize_predictions(
        net,
        imgs_dir,
        classes,
        device,
        args,
        out_dir="Experiments",
        save_heatmaps=False,
        images_per_class=None,
        top_k=None,
    )
