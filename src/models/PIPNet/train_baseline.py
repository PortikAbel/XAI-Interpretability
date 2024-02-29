import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

from data.config import DATASETS
from models.PIPNet.util.data import get_img_loader

# Create trainsforms and dataloader
dataset_name = "road_type_detection"
dataset_config = DATASETS[dataset_name]
img_shape = dataset_config["img_shape"]
num_classes = dataset_config["num_classes"]
mean = dataset_config["mean"]
std = dataset_config["std"]

train_dir = dataset_config["train_dir"]
test_dir = dataset_config.get("test_dir", None)
img_loader = get_img_loader(dataset_name)

# Define transforms for the dataset
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_shape, antialias=True),
    transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_shape, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
])
trainset = torchvision.datasets.ImageFolder(
    train_dir, transform=transform_train, loader=img_loader)
testset = torchvision.datasets.ImageFolder(
    test_dir, transform=transform_test, loader=img_loader
)

# Define the device for training
gpu_id = 2
device = torch.device(f"cuda:{gpu_id}")

# Load pre-trained ResNet-18 model
model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Changing the output layer for the number of classes

# Modify the first layer to accept single-channel images
conv1_w = model.conv1.weight
conv1_w = torch.sum(conv1_w, dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.conv1.weight.data = conv1_w

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Assuming you have your trainloader and testloader ready
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

tensorboard_writer = SummaryWriter(log_dir="runs/baseline_lr-1e-2")

num_epochs = 15
print(f"Number of training epochs: {num_epochs}")
print(f"Len of trainloader: {len(trainloader)}")
print(f"Len of testloader: {len(testloader)}")

# Training the model
for epoch in range(num_epochs):
    print(f"#"*50)
    print(f"Epoch: {epoch}")
    model.train()
    running_loss = 0.0
    running_loss_epoch = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        tensorboard_writer.add_scalar("Loss_train", loss.item(), epoch*len(trainloader)+i)

        running_loss += loss.item()
        running_loss_epoch += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            # tensorboard_writer.add_scalar("Loss_train_100_iter", running_loss/100., epoch*len(trainloader)+i)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    tensorboard_writer.add_scalar("Loss_train_epoch", running_loss_epoch/len(trainloader), epoch)
    
    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    running_eval_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_eval_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    tensorboard_writer.add_scalar("Loss_test_epoch", running_eval_loss/len(testloader), epoch)
    acc = 100 * correct / total
    tensorboard_writer.add_scalar("Test_accuracy", acc, epoch)
    print('Accuracy of the network on the test images: %d %%' % (
        acc))
print('Finished Training')


