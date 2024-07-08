# Exploring the Impact of Backbone Architecture on Explainable CNNs' Interpretability

The present repository contains the code for the experiments described in the paper
```
Ábel Portik, Adél Bajcsi, Annamária Szenkovits, Zalán Bodó. Exploring the Impact of Backbone Architecture on Explainable CNNs' Interpretability.
Submitted to ACTA UNIVERSITATIS SAPIENTIAE, Informatica, 2024.
```

## ProtoPNet

## PIP-Net

## Post-hoc methods

## Evaluation

The evaluation script (`run.py`) resides in the `evaluation` subfolder. 
It is based on the [funnybirds-framework](https://github.com/visinf/funnybirds-framework) repository, therefore we recommend to consult
their readme description before running our code.

The main parameters are the following (for the full list please see the code of [run.py](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/evaluation/run.py)):
* model - name of the model to be used (e.g. PIP-Net, post_hoc, etc.)
* explainer - the explainer model applied (e.g. PIP-Net, GradCam, etc.)
* checkpoint_path - path to the model checkpoint uder evaluation
Parameters for interpretability metrics (for the definition of these see the paper [FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods](https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html)):
* accuracy
* background_independence
* controlled_synthetic_data_check
* single_deletion
* preservation_check
* deletion_check
* target_sensitivity
* distractibility

An example:
```
python3 run.py --data_path data/FunnyBirds \
	--model post_hoc \
	--backbone resnet18 \
	--checkpoint_path models/model_resnet18 \
	--explainer GradCam \
	--accuracy --controlled_synthetic_data_check --single_deletion --preservation_check --deletion_check --distractibility --background_independence --target_sensitivity \
	--gpu 0
```