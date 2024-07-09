# Exploring the Impact of Backbone Architecture on Explainable CNNs' Interpretability

The present repository contains the code for the experiments described in the paper

> Ábel Portik, Adél Bajcsi, Annamária Szenkovits, Zalán Bodó. 
> Exploring the Impact of Backbone Architecture on Explainable CNNs' Interpretability.
> Submitted to ACTA UNIVERSITATIS SAPIENTIAE, Informatica, 2024.

## Explainable models

In the current experiments two explainable models were used: ProtoPNet [^Chen-2018-ProtoPNet]
and PIP-Net [^Nauta-2023-PIP-Net]. The code for these models can be found in the `models` subfolder.
These models are built on top of a CNN backbone, which can be any of the following: `ResNet18`, 
`ResNet34`, `ResNet50`, `VGG11`, `VGG13`, `VGG16`, `VGG19`, and `ConvNeXt-Tiny`. 

Both models are train using [src/scripts/train/main.py](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/scripts/train/main.py) script.

### ProtoPNet

To train a ProtoPNet model the `--model ProtoPNet` must be set to the above-mentioned script.
For the full list of parameters see the [README](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/models/ProtoPNet/README.md) 
in the `models/ProtoPNet` subfolder.

An example:
```shell
python3 main.py --model ProtoPNet --net vgg11 --dataset Funny --image_width 224 --batch_size 64 --gpu_ids 0 --enable_console --disable_normalize
```

### PIP-Net

PIP-Net is a model extending ProtoPNet. To train a PIP-Net model the `--model PIPNet` 
must be set to the above-mentioned script. For the full list of parameters see the 
[README](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/models/PIPNet/README.md) in the `models/PIPNet` subfolder.

An example:
```shell
python3 main.py --model PIPNet --net vgg11 --dataset Funny
```

## Post-hoc methods

Post-hoc methods (e.g. GradCAM [^Selvaraju-2016-GradCam]) can use any CNN backbone (`models/post_hoc`). 
The script [train_posthoc.py](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/models/post_hoc/train_posthoc.py) 
can be used to train such CNN backbones, and at the present supports the architectures 
`ResNet18`, `ResNet34`, `ResNet50`, `VGG11`, `VGG13`, `VGG16`, `VGG19` and 
`ConvNeXt-Tiny`, however, it can be easily extended (in this case the evaluation 
script has to be extended as well).

An example:
```shell
python3 train_posthoc.py --backbone resnet18 --dataset Funny
```

## Evaluation

The evaluation script ([run.py](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/evaluation/run.py)) resides in the `evaluation` subfolder. 
It is based on the [funnybirds-framework](https://github.com/visinf/funnybirds-framework) repository, therefore we recommend to consult
their readme description before running our code.

The main parameters are the following (for the full list please see the code of [run.py](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/evaluation/run.py)):
* model - name of the model to be used (e.g. PIP-Net, post_hoc, etc.)
* explainer - the explainer model applied (e.g. PIP-Net, GradCam, etc.)
* checkpoint_path - path to the model checkpoint uder evaluation

Parameters for interpretability metrics (for the definition of these see the paper [FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods](https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html)):
* _accuracy_
* _background_independence_
* _controlled_synthetic_data_check_
* _single_deletion_
* _preservation_check_
* _deletion_check_
* _target_sensitivity_
* _distractibility_

An example:
```shell
python3 run.py --data_path data/FunnyBirds \
	--model post_hoc \
	--backbone resnet18 \
	--checkpoint_path models/model_resnet18 \
	--explainer GradCam \
	--accuracy --controlled_synthetic_data_check --single_deletion --preservation_check --deletion_check --distractibility --background_independence --target_sensitivity \
	--gpu 0
```

# References:

[^Selvaraju-2016-GradCam]: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2016).
  Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. 
  [doi:10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7)

[^Chen-2018-ProtoPNet]: Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019).
  This looks like that: deep learning for interpretable image recognition. 
  Advances in neural information processing systems, 32.
  [doi:10.48550/arXiv.1806.10574](https://doi.org/10.48550/arXiv.1806.10574)

[^Nauta-2023-PIP-Net]: Nauta, M., Schlötterer, J., Van Keulen, M., & Seifert, C. (2023). 
  Pip-net: Patch-based intuitive prototypes for interpretable image classification. 
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2744-2753).
  [doi:10.1109/CVPR52729.2023.00269](https://doi.org/10.1109/CVPR52729.2023.00269)