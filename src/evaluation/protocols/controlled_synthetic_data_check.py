from tqdm import tqdm
from torch.utils.data import DataLoader
from data.funny_birds import FunnyBirds


def controlled_synthetic_data_check_protocol(model, explainer, args):
    transforms = None

    test_dataset = FunnyBirds(
        args.data, "test", get_part_map=True, transform=transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    mcsdc_for_thresholds = {}
    for threshold in thresholds:
        mcsdc_for_thresholds[threshold] = 0
    number_valid_samples = 0
    for samples in tqdm(test_loader):
        images = samples["image"]
        target = samples["class_idx"]
        part_maps = samples["part_map"]
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            part_maps = part_maps.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # make sure that model correctly classifies instance
        output = model(images)
        if output.argmax(1) != target:
            continue

        important_parts_for_thresholds = explainer.get_important_parts(
            images,
            part_maps,
            target,
            test_dataset.colors_to_part,
            thresholds=thresholds,
        )
        for important_parts, threshold in zip(
            important_parts_for_thresholds, thresholds
        ):
            minimal_sufficient_part_sets = (
                test_dataset.get_minimal_sufficient_part_sets(target[0].item())
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
