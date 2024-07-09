# Train ProtoPNet

To train a ProtoPNet model the `--model ProtoPNet` must be set to the above-mentioned script.
For the full list of parameters see the [README](https://github.com/PortikAbel/XAI-Interpretability/blob/main/src/models/ProtoPNet/README.md) 
in the `models/ProtoPNet` subfolder.

An example:
```shell
python3 main.py --model ProtoPNet --net vgg11 --dataset Funny --image_width 224 --batch_size 64 --gpu_ids 0 --enable_console --disable_normalize
```

## Usage: 

```
Train ProtoPNet [-h] [--seed SEED] [--num_workers NUM_WORKERS]
                       [--gpu_ids GPU_IDS] [--disable_gpu] [--log_dir LOG_DIR]
                       [--dir_for_saving_images DIR_FOR_SAVING_IMAGES]
                       [--save_all_models] [--dataset DATASET]
                       [--validation_size VALIDATION_SIZE]
                       [--disable_normalize] [--image_width IMAGE_WIDTH]
                       [--image_height IMAGE_HEIGHT]
                       [--net NET | --state_dict_dir_net STATE_DICT_DIR_NET | --backbone_only]
                       [--batch_size BATCH_SIZE]
                       [--batch_size_push BATCH_SIZE_PUSH] [--epochs EPOCHS]
                       [--epochs_warm EPOCHS_WARM]
                       [--epochs_finetune EPOCHS_FINETUNE]
                       [--push_start PUSH_START]
                       [--push_interval PUSH_INTERVAL]
                       [--n_prototypes_per_class N_PROTOTYPES_PER_CLASS]
                       [--prototype_depth PROTOTYPE_DEPTH]
                       [--prototype_activation_function PROTOTYPE_ACTIVATION_FUNCTION]
                       [--add_on_layers_type {regular,bottleneck}]
                       [--disable_pretrained] [--bias] [--optimizer OPTIMIZER]
                       [--warm_lr_add_on_layers WARM_LR_ADD_ON_LAYERS]
                       [--warm_lr_prototype_vectors WARM_LR_PROTOTYPE_VECTORS]
                       [--joint_lr_features JOINT_LR_FEATURES]
                       [--joint_lr_add_on_layers JOINT_LR_ADD_ON_LAYERS]
                       [--joint_lr_prototype_vectors JOINT_LR_PROTOTYPE_VECTORS]
                       [--joint_lr_step JOINT_LR_STEP]
                       [--finetune_lr FINETUNE_LR]
                       [--weight_decay WEIGHT_DECAY] [--weighted_loss]
                       [--separation_type {max,avg,margin}]
                       [--binary_cross_entropy]
                       [--coefficient_cross_entropy COEFFICIENT_CROSS_ENTROPY]
                       [--coefficient_clustering COEFFICIENT_CLUSTERING]
                       [--coefficient_separation COEFFICIENT_SEPARATION]
                       [--coefficient_separation_margin COEFFICIENT_SEPARATION_MARGIN]
                       [--coefficient_l1 COEFFICIENT_L1]
                       [--coefficient_l2 COEFFICIENT_L2]
                       [--prototype_img_filename_prefix PROTOTYPE_IMG_FILENAME_PREFIX]
                       [--prototype_self_act_filename_prefix PROTOTYPE_SELF_ACT_FILENAME_PREFIX]
                       [--proto_bound_boxes_filename_prefix PROTO_BOUND_BOXES_FILENAME_PREFIX]
                       [--weight_matrix_filename WEIGHT_MATRIX_FILENAME]
                       [--visualize_topk] [--visualize_predictions]
```

## Options:
Necessary parameters to train a ProtoPNet

> `-h, --help`      show this help message and exit

> `--seed SEED`     Random seed. Note that there will still be differences
                    between runs due to nondeterminism. See [here](https://pytorch.org/docs/stable/notes/randomness.html).

> `--num_workers NUM_WORKERS`
                    Num workers in dataloaders.

> `--image_width IMAGE_WIDTH`
                    The width of the images in the dataset

> `--image_height IMAGE_HEIGHT`
                    The height of the images in the dataset

> `--net NET`       Base network used as backbone of ProtoPNet. Default is
                    resnet18. Options are: resnet18, resnet34, resnet50,
                    resnet50_inat, resnet101, resnet152, vgg13, vgg16 and
                    vgg19.

> `--state_dict_dir_net STATE_DICT_DIR_NET`
                    The directory containing a state dict with a
                    pretrained ProtoPNet. E.g.,
                    ./runs/ProtoPNet/<run_name>/checkpoints/net_pretrained
--backbone_only       Flag that indicates whether to train only the backbone
                    network.

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

> `--prototype_img_filename_prefix PROTOTYPE_IMG_FILENAME_PREFIX`
                        Prefix for the prototype images.

> `--prototype_self_act_filename_prefix PROTOTYPE_SELF_ACT_FILENAME_PREFIX`
                    Prefix for the prototype self activations.

> `--proto_bound_boxes_filename_prefix PROTO_BOUND_BOXES_FILENAME_PREFIX`
                    Prefix for the prototype images with bounding box.

> `--weight_matrix_filename WEIGHT_MATRIX_FILENAME`
                    Filename for the weight matrix.

### Dataset:
Specifies the dataset to use and its hyperparameters

> `--dataset DATASET`     Data set on ProtoPNet should be trained

> `--validation_size VALIDATION_SIZE`
                    Split between training and validation set. Can be zero
                    when there is a separate test or validation directory.
                    Should be between 0 and 1. Used for partimagenet (e.g.
                    0.2)

> `--disable_normalize`   Flag that disables normalization of the images

### Network parameters:
Specifies the used network's hyperparameters

> `--batch_size BATCH_SIZE`
                    Batch size when training the model using minibatch
                    gradient descent. Batch size is multiplied with number
                    of available GPUs

> `--batch_size_push BATCH_SIZE_PUSH`
                    Batch size when pushing the prototypes to the feature
                    space

> `--epochs EPOCHS` The number of epochs ProtoPNet should be trained
                    (second training stage)

> `--epochs_warm EPOCHS_WARM`
                    Number of epochs to pre-train the prototypes (first
                    training stage). Recommended to train at least until
                    the align loss < 1

> `--epochs_finetune EPOCHS_FINETUNE`
                    During fine-tuning, only train classification layer
                    and freeze rest. Usually done for a few epochs (at
                    least 1, more depends on size of dataset)

> `--add_on_layers_type {regular,bottleneck}`
                    Type of add-on layer to use.

> `--disable_pretrained`  
                    When set, the backbone network is initialized with
                    random weights instead of being pretrained on another
                    dataset).

> `--bias`          Flag that indicates whether to include a trainable
                    bias in the linear classification layer.

### Optimizer:
Specifies the optimizer to use and its hyperparameters

> `--optimizer OPTIMIZER`    
                    The optimizer that should be used when training
                    ProtoPNet

> `--weight_decay WEIGHT_DECAY`  
                    Weight decay used in the optimizer

### Loss:
Specifies the loss function to use and its hyperparameters

> `--weighted_loss` Flag that weights the loss based on the class balance
                    of the dataset. Recommended to use when data is
                    imbalanced.

> `--separation_type {max,avg,margin}`
                    Type of separation loss to use.

> `--binary_cross_entropy`
                    Flag that indicates whether to use binary cross
                    entropy loss.

### Visualization:
Specifies which visualizations should be generated

> `--visualize_topk`
>                   Flag that indicates whether to visualize the top k
                    activations of each prototype from test set.

> `--visualize_predictions`
                    Flag that indicates whether to visualize the
                    predictions on test data and the learned prototypes.