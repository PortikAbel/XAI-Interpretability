from torch.utils.data import DataLoader
from tqdm import tqdm

from data.funny_birds import FunnyBirds


def preservation_check_protocol(model, explainer, args):
    transforms = None

    test_dataset = FunnyBirds(
        args.data_dir, "test", get_part_map=True, transform=transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []

    number_valid_samples = 0
    for samples in tqdm(test_loader):
        images = samples["image"]
        part_maps = samples["part_map"]
        class_name = samples["class_name"].item()
        image_idx = samples["image_idx"].item()
        if not args.disable_gpu:
            images = images.cuda(args.device_ids[0], non_blocking=True)
            part_maps = part_maps.cuda(args.device_ids[0], non_blocking=True)

        output = model(images)
        model_prediction_original = output.argmax(1)

        important_parts_for_thresholds = explainer.get_important_parts(
            images,
            part_maps,
            model_prediction_original,
            test_dataset.colors_to_part,
            thresholds=thresholds,
        )
        for important_parts, threshold in zip(
            important_parts_for_thresholds, thresholds
        ):
            all_parts = list(test_dataset.parts.keys())
            parts_removed = list(set(all_parts) - set(important_parts))

            image2 = test_dataset.get_intervention(
                class_name, image_idx, parts_removed
            )["image"]
            image2 = image2.cuda(args.device_ids[0], non_blocking=True)
            output2 = model(image2)
            model_prediction_removed = output2.argmax(1)

            if model_prediction_original == model_prediction_removed:
                scores_for_thresholds[threshold].append(1.0)
            else:
                scores_for_thresholds[threshold].append(0.0)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        scores_for_thresholds[threshold] = sum(scores_for_thresholds[threshold]) / len(
            scores_for_thresholds[threshold]
        )

    print("Preservation Check Score: ", scores_for_thresholds)
    return scores_for_thresholds
