import torch
from tqdm import tqdm

def target_sensitivity_protocol(model, dataloader, explainer, args):
    def class_overlap(parts1, parts2):
        overlap_parts = []
        for key in parts1.keys():
            if parts1[key] == parts2[key]:
                overlap_parts.append(key)
        return overlap_parts

    target_sensitivity_score = []
    number_valid_samples = 0
    number_assumption_wrong = 0

    assumption_strengths = []

    for sample in tqdm(dataloader):
        image = sample["image"]
        target = sample["target"]
        part_map = sample["part_map"]
        params = sample["params"]
        class_name = sample["class_name"].item()
        image_idx = sample["image_idx"].item()
        params = dataloader.dataset.get_params_for_single(params)
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            part_map = part_map.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(image)

        # get two classes that have each 2 parts in common with current target class,
        # i.e. 3 parts distance
        classes_w_two_overlap = dataloader.dataset.get_classes_with_distance_n(target[0], 3)
        # get two classes out of these that don't have overlap in the two parts
        # that overlap with target class.
        # E.g. one overlaps in foot and beak and the other in tail and wing
        found_classes = False
        for class1_idx in range(len(classes_w_two_overlap)):
            for class2_idx in range(class1_idx + 1, len(classes_w_two_overlap)):
                class1 = classes_w_two_overlap[class1_idx]
                class2 = classes_w_two_overlap[class2_idx]
                parts_target = dataloader.dataset.classes[target[0]]["parts"]
                parts_class1 = dataloader.dataset.classes[class1]["parts"]
                parts_class2 = dataloader.dataset.classes[class2]["parts"]
                overlap_target_class1 = class_overlap(parts_target, parts_class1)
                overlap_target_class2 = class_overlap(parts_target, parts_class2)

                if set(overlap_target_class1).isdisjoint(set(overlap_target_class2)):
                    found_classes = True
                    break
            if found_classes:
                break

        class1 = torch.tensor([class1]).cuda(args.gpu, non_blocking=True)
        class2 = torch.tensor([class2]).cuda(args.gpu, non_blocking=True)

        # skip sample if assumption does not hold
        # class a: removing A parts should result in larger drop than removing B parts
        #   and removing B parts should result in larger increase than removing A parts
        # class b: removing B parts should result in larger drop than removing A parts

        image2 = dataloader.dataset.get_intervention(
            class_name, image_idx, overlap_target_class1
        )["image"]
        image2 = image2.cuda(args.gpu, non_blocking=True)
        output_wo_parts_from_class1 = model(image2)

        image2 = dataloader.dataset.get_intervention(
            class_name, image_idx, overlap_target_class2
        )["image"]
        image2 = image2.cuda(args.gpu, non_blocking=True)
        output_wo_parts_from_class2 = model(image2)

        drop_class1_when_rm_class1_parts = (
            output_wo_parts_from_class1[0][class1] - output[0][class1]
        )
        drop_class1_when_rm_class2_parts = (
            output_wo_parts_from_class2[0][class1] - output[0][class1]
        )

        drop_class2_when_rm_class1_parts = (
            output_wo_parts_from_class1[0][class2] - output[0][class2]
        )
        drop_class2_when_rm_class2_parts = (
            output_wo_parts_from_class2[0][class2] - output[0][class2]
        )

        # smaller because the drop should be more negative
        if not (
            drop_class1_when_rm_class1_parts < drop_class1_when_rm_class2_parts
            and drop_class2_when_rm_class2_parts < drop_class2_when_rm_class1_parts
        ):
            number_assumption_wrong += 1
            continue

        assumption_strengths.append(
            drop_class1_when_rm_class2_parts.item()
            - drop_class1_when_rm_class1_parts.item()
        )

        part_importances_class1 = explainer.get_part_importance(
            image, part_map, class1, dataloader.dataset.colors_to_part
        )
        part_importances_class2 = explainer.get_part_importance(
            image, part_map, class2, dataloader.dataset.colors_to_part
        )

        overlap_target_class1_importance_class1 = 0
        overlap_target_class1_importance_class2 = 0
        for part in overlap_target_class1:
            overlap_target_class1_importance_class1 += part_importances_class1[part]
            overlap_target_class1_importance_class2 += part_importances_class2[part]

        if (
            overlap_target_class1_importance_class1
            > overlap_target_class1_importance_class2
        ):
            target_sensitivity_score.append(1.0)
        else:
            target_sensitivity_score.append(0.0)

        overlap_target_class2_importance_class1 = 0
        overlap_target_class2_importance_class2 = 0
        for part in overlap_target_class2:
            overlap_target_class2_importance_class1 += part_importances_class1[part]
            overlap_target_class2_importance_class2 += part_importances_class2[part]

        if (
            overlap_target_class2_importance_class1
            < overlap_target_class2_importance_class2
        ):
            target_sensitivity_score.append(1.0)
        else:
            target_sensitivity_score.append(0.0)

        number_valid_samples += 1
        if args.nr_itrs == number_valid_samples:
            break

    target_sensitivity_score = sum(target_sensitivity_score) / len(
        target_sensitivity_score
    )
    print("Number of filtered samples:", number_assumption_wrong)
    print("Number of valid samples:", number_valid_samples)
    print("Target Sensitivity Score: ", target_sensitivity_score)
    return target_sensitivity_score
