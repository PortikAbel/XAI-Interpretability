# Train PIP-Net

PIP-Net is a model extending ProtoPNet.

An example:
```shell
python3 main.py --model PIPNet --net vgg11 --dataset Funny
```

## Usage: 
```
Train PIPNet [-h] [--seed SEED] [--num_workers NUM_WORKERS]
                    [--gpu_ids GPU_IDS] [--disable_gpu] [--log_dir LOG_DIR]
                    [--dir_for_saving_images DIR_FOR_SAVING_IMAGES]
                    [--save_all_models] [--dataset DATASET]
                    [--validation_size VALIDATION_SIZE] [--disable_normalize]
                    [--image_width IMAGE_WIDTH] [--image_height IMAGE_HEIGHT]
                    [--net NET | --state_dict_dir_net STATE_DICT_DIR_NET]
                    [--batch_size BATCH_SIZE]
                    [--batch_size_pretrain BATCH_SIZE_PRETRAIN]
                    [--train_backbone_during_pretrain] [--epochs EPOCHS]
                    [--epochs_pretrain EPOCHS_PRETRAIN]
                    [--freeze_epochs FREEZE_EPOCHS]
                    [--epochs_finetune EPOCHS_FINETUNE]
                    [--num_features NUM_FEATURES] [--disable_pretrained]
                    [--bias] [--optimizer OPTIMIZER] [--lr LR]
                    [--lr_block LR_BLOCK] [--lr_net LR_NET]
                    [--weight_decay WEIGHT_DECAY] [--weighted_loss]
                    [--tanh_loss TANH_LOSS] [--unif_loss UNIF_LOSS]
                    [--variance_loss VARIANCE_LOSS]
                    [--log_prototype_activations_violin_plot]
                    [--visualize_topk] [--visualize_predictions]
                    [--evaluate_purity] [--evaluate_ood]
                    [--extra_test_image_folder EXTRA_TEST_IMAGE_FOLDER]
```


## Options:
Necessary parameters to train a PIPNet
>   `-h`, `--help`  show this help message and exit

> `--seed SEED`     Random seed. Note that there will still be differences
                    between runs due to nondeterminism. See
                    https://pytorch.org/docs/stable/notes/randomness.html

> `--num_workers NUM_WORKERS`
                    Num workers in dataloaders.

> `--image_width IMAGE_WIDTH`
                    The width of the images in the dataset

> `--image_height IMAGE_HEIGHT`
                    The height of the images in the dataset

> `--net NET`        Base network used as backbone of PIP-Net. Default is
                    convnext_tiny_26 with adapted strides to output 26x26
                    latent representations. Other option is
                    convnext_tiny_13 that outputs 13x13 (smaller and
                    faster to train, less fine-grained). Pretrained
                    network on iNaturalist is only available for
                    resnet50_inat. Options are: resnet18, resnet34,
                    resnet50, resnet50_inat, resnet101, resnet152,
                    convnext_tiny_26 and convnext_tiny_13.

> `--state_dict_dir_net STATE_DICT_DIR_NET`
                    The directory containing a state dict with a
                    pretrained PIP-Net. E.g., ./code/PIPNet/runs/run_pipne
                    t/checkpoints/net_pretrained

### GPU:
Specifies the GPU settings

> `--gpu_ids GPU_IDS`     ID of gpu. Can be separated with comma

> `--disable_gpu`         Flag that disables GPU usage if set

### Logging:
Specifies the directory where the log files and other outputs should be
saved

> `--log_dir LOG_DIR`     The directory in which train progress should be logged

> `--dir_for_saving_images DIR_FOR_SAVING_IMAGES`
                    Directory for saving the prototypes and explanations

> `--save_all_models`     Flag to save the model in each epoch

### Dataset:
Specifies the dataset to use and its hyperparameters

> `--dataset DATASET`     Data set on ProtoPNet should be trained

> `--validation_size VALIDATION_SIZE`
                    Split between training and validation set. Can be zero
                    when there is a separate test or validation directory.
                    Should be between 0 and 1. Used for partimagenet (e.g.
                    0.2)

> `--disable_normalize`   Flag that disables normalization of the images

##### Network parameters:
Specifies the used network's hyperparameters

> `--batch_size BATCH_SIZE`
                      Batch size when training the model using minibatch
                      gradient descent. Batch size is multiplied with number
                      of available GPUs

> `--batch_size_pretrain BATCH_SIZE_PRETRAIN`
                      Batch size when pretraining the prototypes (first
                      training stage)

> `--train_backbone_during_pretrain`
                      To train the whole backbone during pretrain (e.g. if
                      dataset is very different from ImageNet)

> `--epochs EPOCHS`       The number of epochs PIP-Net should be trained (second
                      training stage)

> `--epochs_pretrain EPOCHS_PRETRAIN`
                      Number of epochs to pre-train the prototypes (first
                      training stage). Recommended to train at least until
                      the align loss < 1

> `--freeze_epochs FREEZE_EPOCHS`
                      Number of epochs where pretrained features_net will be
                      frozen while training classification layer (and last
                      layer(s) of backbone)

> `--epochs_finetune EPOCHS_FINETUNE`
                      During fine-tuning, only train classification layer
                      and freeze rest. Usually done for a few epochs (at
                      least 1, more depends on size of dataset)

> `--num_features NUM_FEATURES`
                      Number of prototypes. When zero (default) the number
                      of prototypes is the number of output channels of
                      backbone. If this value is set, then a 1x1 conv layer
                      will be added. Recommended to keep 0, but can be
                      increased when number of classes > num output channels
                      in backbone.

> `--disable_pretrained`  When set, the backbone network is initialized with
                      random weights instead of being pretrained on another
                      dataset).

> `--bias`                Flag that indicates whether to include a trainable
                      bias in the linear classification layer.

##### Optimizer:
Specifies the optimizer to use and its hyperparameters

> `--optimizer OPTIMIZER`
                      The optimizer that should be used when training PIP-
                      Net

> `--lr LR`           The optimizer learning rate for training the weights
                      from prototypes to classes

> `--lr_block LR_BLOCK`   The optimizer learning rate for training the last
                      convolutional layers of the backbone

> `--lr_net LR_NET`   The optimizer learning rate for the backbone. Usually
                      similar as lr_block.

> `--weight_decay WEIGHT_DECAY`
                      Weight decay used in the optimizer

##### Loss:
Specifies the loss function to use and its hyperparameters

> `--weighted_loss`   Flag that weights the loss based on the class balance
                      of the dataset. Recommended to use when data is
                      imbalanced.

> `--tanh_loss TANH_LOSS`
                      tanh loss regulates that every prototype should be at
                      least once present in a mini-batch.

> `--unif_loss UNIF_LOSS`
                      Our tanh-loss optimizes for uniformity and was
                      sufficient for our experiments. However, if
                      pretraining of the prototypes is not working well for
                      your dataset, you may try to add another uniformity
                      loss from https://www.tongzhouwang.info/hypersphere/

> `--variance_loss VARIANCE_LOSS`
                      Regularizer term that enforces variance of features
                      from https://arxiv.org/abs/2105.04906

##### Visualization:
Specifies which visualizations should be generated

> `--visualize_topk`  Flag that indicates whether to visualize the top k
                      activations of each prototype from test set.

> `--visualize_predictions`
                      Flag that indicates whether to visualize the
                      predictions on test data and the learned prototypes.

##### Evaluation:
Specifies which evaluation metrics should be calculated

> `--evaluate_purity` Flag that indicates whether to evaluate purity of
                      prototypes. Prototype purity is a metric for measuring
                      the overlap between the position of learned prototypes
                      and labeled feature centers in the image space.
                      Currently is measurable only on CUB-200-2011.

> `--evaluate_ood`    Flag that indicates whether to evaluate OoD detection
                      on other datasets than train set.

> `--extra_test_image_folder EXTRA_TEST_IMAGE_FOLDER`
                      Folder with images that PIP-Net will predict and
                      explain, that are not in the training or test set.
                      E.g. images with 2 objects or OOD image. Images should
                      be in subfolder. E.g. images in ./experiments/images/,
                      and argument --./experiments