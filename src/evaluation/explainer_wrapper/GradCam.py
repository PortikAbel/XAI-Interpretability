import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from evaluation.explainer_wrapper.base import AbstractAttributionExplainer


class GradCamExplainer(AbstractAttributionExplainer):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # self.dilation = nn.MaxPool2d(5, stride=1, padding=2)
        self.dilation = nn.MaxPool2d(1, stride=1, padding=0)

    def explain(self, input, target):
        targets = [ClassifierOutputTarget(target)]
        with GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
        # with GradCAMPlusPlus(model=self.model, target_layers=[self.target_layer]) as cam:
        # with AblationCAM(model=self.model, target_layers=[self.target_layer]) as cam:
        # with EigenCAM(model=self.model, target_layers=[self.target_layer]) as cam:
        # with EigenGradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
        # with HiResCAM(model=self.model, target_layers=[self.target_layer]) as cam:
        # with XGradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            grayscale_cams = cam(input_tensor=input, targets=targets)
            
        return torch.from_numpy(grayscale_cams).to(input.device)
