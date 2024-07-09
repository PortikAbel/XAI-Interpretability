import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.post_hoc.util.args import get_args

import torchvision
import torchvision.transforms as transforms

from data.config import DATASETS

from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg11, vgg13, vgg16, vgg19
from models.convnext import convnext_tiny

import torch.optim as optim

# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def run_nn(args=None):
    # Define the device for training
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")

    # CUDA device info:
    if device.type == "cuda":
        print(
            "#"*50 + "\nDevice name: {}\nMemory Usage:\nAllocated: {} GB\n"
                     "Cached: {} GB".format(
                    torch.cuda.get_device_name(0),
                          round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1),
                          round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                     ) + "\n" + "#" * 50)
    
    # Create transforms and dataloader
    dataset_name = args.dataset
    
    dataset_config = DATASETS[dataset_name]
    num_classes = dataset_config["num_classes"]

    # Define transforms for the dataset
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_set = torchvision.datasets.ImageFolder(
        dataset_config["train_dir"], transform=transform_train
    )
    test_set = torchvision.datasets.ImageFolder(
        dataset_config["test_dir"], transform=transform_test
    )

    # Dataloader:
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False
    )

    pretrained = not args.disable_pretrained

    print(args.backbone)

    match args.backbone:
        case "resnet18":
            net = resnet18(num_classes=num_classes, pretrained=pretrained)
        case "resnet34":
            net = resnet34(num_classes=num_classes, pretrained=pretrained)
        case "resnet50":
            net = resnet50(num_classes=num_classes, pretrained=pretrained)
        case "vgg11":
            net = vgg11(num_classes=num_classes, pretrained=pretrained)
        case "vgg13":
            net = vgg13(num_classes=num_classes, pretrained=pretrained)
        case "vgg16":
            net = vgg16(num_classes=num_classes, pretrained=pretrained)
        case "vgg19":
            net = vgg19(num_classes=num_classes, pretrained=pretrained)
        case "convnext":
            net = convnext_tiny(num_classes=num_classes, pretrained=pretrained)
        case _:
            raise NotImplementedError(
                f"Standard network {args.backbone!r} not implemented"
            )

    net = net.to(device=device)

    # tensorboard:
    tensorboard_writer = SummaryWriter(log_dir="runs/" + args.backbone)

    if os.path.exists("models/model_" + args.backbone): 
        # Load model if exists:
        checkpoint = torch.load("models/model_" + args.backbone)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()

    else:
        # Define loss function and optimizer:
        criterion = nn.CrossEntropyLoss()
        match args.optimizer:
            case "Adam":
                optimizer = optim.Adam(net.parameters())  # resnet
            case "SGD":
                optimizer = optim.SGD(net.parameters())  # vgg, convnext
            case _:
                raise NotImplementedError(
                    f"Optimizer {args.optimizer!r} not implemented"
                )

        epoch_last = args.last_epoch
        if os.path.exists(f"models/model_{args.backbone}_{epoch_last}"):
            checkpoint = torch.load(f"models/model_{args.backbone}_{epoch_last}")
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("...LOADED...")
        else:
            epoch_last = -1

        num_epochs = args.epochs

        print(f"Number of training epochs: {num_epochs}")
        print(f"Len of train_loader: {len(train_loader)}")
        print(f"Len of test_loader: {len(test_loader)}")
        print(f"Number of classes: {num_classes}")

        step = 100

        # Training the model:
        for epoch in range(epoch_last+1, num_epochs):
            print("#" * 50)
            print(f"Epoch: {epoch}")
            net.train()
            running_loss = 0.0
            running_loss_epoch = 0.0
            total_correct = 0
            total_samples = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # tensorboard_writer.add_scalar(
                #   "Loss_train", loss.item(), epoch * len(train_loader) + i
                # )

                running_loss += loss.item()
                running_loss_epoch += loss.item()
                
                if i % step == step-1:  # print
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate the training accuracy for this epoch:
            accuracy = total_correct / total_samples
            print(f"Train accuracy: {accuracy:.2%}")

            tensorboard_writer.add_scalar(
                "Loss_train_epoch", running_loss_epoch / len(train_loader), epoch
            )

            if (epoch+1) % args.save_step == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                }, "models/model_" + args.backbone + "_" + str(epoch))

            # Evaluate the model on the test set:
            net.eval()
            correct = 0
            total = 0
            running_eval_loss = 0.0

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    running_eval_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total
            print(f"Test accuracy: {acc:.2%}")

            tensorboard_writer.add_scalar("Loss_test_epoch", running_eval_loss / len(test_loader), epoch)
            # tensorboard_writer.add_scalar("Test_accuracy", acc, epoch)

        print("Finished Training")

        # Save model:
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
            }, "models/model_" + args.backbone)
    ##############################################################

    # Evaluate the model on the test set:
    correct = 0
    total = 0
    running_eval_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"Accuracy of the network on the test images: {acc:.2%}")

    ##############################################################


if __name__ == "__main__":
    args = get_args()

    # print(args)

    run_nn(args)
