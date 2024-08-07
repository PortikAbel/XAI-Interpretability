from scipy import stats
from tqdm import tqdm


def single_deletion_protocol(model, dataloader, explainer, args, log):

    correlations = []
    number_valid_samples = 0
    for sample in tqdm(dataloader):
        image = sample["image"]
        target = sample["target"]
        part_map = sample["part_map"]
        params = sample["params"]
        class_name = sample["class_name"].item()
        image_idx = sample["image_idx"].item()
        params = dataloader.dataset.get_params_for_single(params)
        if not args.disable_gpu:
            image = image.cuda(args.device_ids[0], non_blocking=True)
            part_map = part_map.cuda(args.device_ids[0], non_blocking=True)
            target = target.cuda(args.device_ids[0], non_blocking=True)

        score = {}

        output = model(image)
        original_score = output[0, target].item()

        # get scores for removed parts
        bird_parts_keys = list(dataloader.dataset.parts.keys())

        for remove_part in bird_parts_keys:
            image2 = dataloader.dataset.get_intervention(
                class_name, image_idx, [remove_part]
            )["image"]

            image2 = image2.cuda(args.device_ids[0], non_blocking=True)
            output = model(image2)

            score[remove_part.split("_")[0]] = output[
                0, target
            ].item()  # only keep part name, i.e. eye, instead of eye_model

        part_importances = explainer.get_part_importance(
            image, part_map, target, dataloader.dataset.colors_to_part
        )
        score_diffs = {}
        for score_key in score.keys():
            score_diffs[score_key] = original_score - score[score_key]

        # map both in comparable range [-1,1] NOT DONE
        score_diffs_normalized = []
        part_importances_normalized = []
        for key in score_diffs.keys():
            score_diffs_normalized.append(
                score_diffs[key]
            )  # not necessary to normalize with spearmanr coefficient
            part_importances_normalized.append(
                part_importances[key]
            )  # not necessary to normalize with spearmanr coefficient

        correlation, p_value = stats.spearmanr(
            score_diffs_normalized, part_importances_normalized
        )

        import math

        if math.isnan(correlation):
            continue

        correlations.append(correlation * 0.5 + 0.5)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    log.info(
        f"Mean Single Deletion Correlation: {sum(correlations) / len(correlations)}"
    )
    return sum(correlations) / len(correlations)
