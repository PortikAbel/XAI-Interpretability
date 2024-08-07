import re

from tqdm import tqdm


def background_independence_protocol(model, dataloader, args, log):
    total_background_parts = 0
    number_relevant_background_parts = 0

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

        bg_keys = list(filter(lambda x: x.startswith("bg_"), params.keys()))
        bg_object_ids = [int(s) for s in re.findall(r"\b\d+\b", params[bg_keys[0]])]

        for i in range(len(bg_object_ids)):
            image2 = dataloader.dataset.get_background_intervention(
                class_name, image_idx, i
            )["image"]

            image2 = image2.cuda(args.device_ids[0], non_blocking=True)
            output = model(image2)

            score["bg_" + str(i).zfill(3)] = output[0, target].item()

        threshold_for_bg_importances = original_score * 0.05  # 5%

        for score_key in score.keys():
            score_diff = original_score - score[score_key]
            total_background_parts += 1.0

            if (
                abs(score_diff) >= abs(threshold_for_bg_importances)
                and original_score > score[score_key]
            ):
                number_relevant_background_parts += 1.0

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    background_dependence = (
        1 - number_relevant_background_parts / total_background_parts
    )

    log.info(f"Background Dependence Score: {background_dependence}")
    return background_dependence
