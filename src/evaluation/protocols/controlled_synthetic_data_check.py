from tqdm import tqdm

def controlled_synthetic_data_check_protocol(model, dataloader, explainer, args):

    thresholds = explainer.get_p_thresholds()
    mcsdc_for_thresholds = {}
    for threshold in thresholds:
        mcsdc_for_thresholds[threshold] = 0
    number_valid_samples = 0
    for samples in tqdm(dataloader):
        images = samples["image"]
        target = samples["target"]
        part_maps = samples["part_map"]
        if not args.disable_gpu:
            images = images.cuda(args.device_ids[0], non_blocking=True)
            part_maps = part_maps.cuda(args.device_ids[0], non_blocking=True)
            target = target.cuda(args.device_ids[0], non_blocking=True)

        # make sure that model correctly classifies instance
        output = model(images)
        if output.argmax(1) != target:
            continue

        important_parts_for_thresholds = explainer.get_important_parts(
            images,
            part_maps,
            target,
            dataloader.dataset.colors_to_part,
            thresholds=thresholds,
        )
        for important_parts, threshold in zip(
            important_parts_for_thresholds, thresholds
        ):
            minimal_sufficient_part_sets = (
                dataloader.dataset.get_minimal_sufficient_part_sets(target[0].item())
            )
            max_J = 0
            for minimal_sufficient_part_set in minimal_sufficient_part_sets:
                minimal_sufficient_part_set = list(
                    map(
                        lambda part_string: "".join(
                            (x for x in part_string if x.isalpha())
                        ),
                        minimal_sufficient_part_set,
                    )
                )
                J_current = len(
                    set(minimal_sufficient_part_set).intersection(set(important_parts))
                ) / len(minimal_sufficient_part_set)
                if J_current > max_J:
                    max_J = J_current

            mcsdc_for_thresholds[threshold] += max_J
        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        mcsdc_for_thresholds[threshold] = (
            mcsdc_for_thresholds[threshold] / number_valid_samples
        )

    print("mcsdcs: ", mcsdc_for_thresholds)

    return mcsdc_for_thresholds
