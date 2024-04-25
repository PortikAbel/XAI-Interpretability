import re
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.funny_birds import FunnyBirds


def distractibility_protocol(model, explainer, args):
    transforms = None

    # first get scores for different removed parts and original image
    test_dataset = FunnyBirds(
        args.data, "test", get_part_map=True, transform=transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []

    number_valid_samples = 0
    for sample in tqdm(test_loader):
        image = sample["image"]
        target = sample["target"]
        part_map = sample["part_map"]
        params = sample["params"]
        class_name = sample["class_name"].item()
        image_idx = sample["image_idx"].item()
        params = test_dataset.get_params_for_single(params)
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            part_map = part_map.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        score = {}

        output = model(image)
        original_score = output[0, target].item()

        # get scores for removed parts
        bird_parts_keys = list(test_dataset.parts.keys())

        for remove_part in bird_parts_keys:
            image2 = test_dataset.get_intervention(
                class_name, image_idx, [remove_part]
            )["image"]

            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)

            score[remove_part.split("_")[0]] = output[
                0, target
            ].item()  # only keep part name, i.e. eye, instead of eye_model

        bg_keys = list(filter(lambda x: x.startswith("bg_"), params.keys()))
        bg_object_ids = [int(s) for s in re.findall(r"\b\d+\b", params[bg_keys[0]])]

        for i in range(len(bg_object_ids)):
            image2 = test_dataset.get_background_intervention(class_name, image_idx, i)[
                "image"
            ]

            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)

            score["bg_" + str(i).zfill(3)] = output[0, target].item()

        threshold_for_bg_importances = original_score * 0.05  # 5%
        irrelevant_parts = []

        for score_key in score.keys():
            score_diff = original_score - score[score_key]

            if score_diff < 0 or abs(score_diff) < abs(threshold_for_bg_importances):
                irrelevant_parts.append(score_key)

        if len(irrelevant_parts) == 0:
            print("There are no irrelevant parts")
            continue

        explanation_important_parts_for_thresholds = explainer.get_important_parts(
            image,
            part_map,
            target,
            test_dataset.colors_to_part,
            with_bg=True,
            thresholds=thresholds,
        )
        for explanation_important_parts, threshold in zip(
            explanation_important_parts_for_thresholds, thresholds
        ):
            J_current = len(
                set(explanation_important_parts).intersection(set(irrelevant_parts))
            ) / len(irrelevant_parts)
            scores_for_thresholds[threshold].append(J_current)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        scores_for_thresholds[threshold] = 1 - sum(scores_for_thresholds[threshold]) / (
            len(scores_for_thresholds[threshold]) + 1e-8
        )

    print("Mean Distractibility Scores: ", scores_for_thresholds)
    return scores_for_thresholds
