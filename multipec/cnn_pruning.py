import os
from pickle import load
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.nn.utils import prune

from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Global constants
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
SEED = 1

# Paths
INPUT_FOLDER = "data/input/cnn/"
MODEL_PATH = INPUT_FOLDER + 'model.pth'
OUTPUT_FOLDER = "data/output/cnn/"
RESULTS_FOLDER = "data/results/cnn/"
FILENAME_EXTENSIONS = ["_4down", "_34", "_23", "_12", "_05median", "_05up"]

# Set random seed
torch.backends.cudnn.enabled = False
torch.manual_seed(SEED)

# Load MNIST
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
trainset = MNIST('/files/', train=True, download=True, transform=transform)
testset = MNIST('/files/', train=False, download=True, transform=transform)

IDX_TO_LABEL = {v: k for k, v in testset.class_to_idx.items()}
N_CLASSES = len(IDX_TO_LABEL)

# Network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def data_loaders(trainset, testset, trainsize, testsize, NUM_WORKERS=2):
    trainloader = DataLoader(trainset, batch_size=trainsize, num_workers=NUM_WORKERS)
    testloader = DataLoader(testset, batch_size=testsize, num_workers=NUM_WORKERS, shuffle=True)
    return trainloader, testloader

def test_accuracy(model, testloader):
    correct = 0
    labels_sum, output_sum = [], []

    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.data.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.data.view_as(predicted)).sum()
            labels_sum += labels.tolist()
            output_sum += predicted.squeeze().tolist()

    accuracy = correct / len(testloader.dataset)
    f1 = f1_score(labels_sum, output_sum, average=None, zero_division=0)
    precision = precision_score(labels_sum, output_sum, average=None, zero_division=0)
    recall = recall_score(labels_sum, output_sum, average=None, zero_division=0)

    return accuracy, f1, precision, recall

def test_accuracy_per_class(model, testloader):
    correct_pred = {classname: 0 for classname in trainset.classes}
    total_pred = {classname: 0 for classname in trainset.classes}

    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.data.max(1, keepdim=True)[1]

            for label, prediction in zip(labels, predicted):
                label = label.item()
                prediction = prediction.item()
                if label == prediction:
                    correct_pred[trainset.classes[label]] += 1
                total_pred[trainset.classes[label]] += 1

    return {
        classname: (100 * correct_pred[classname] / total_pred[classname]) if total_pred[classname] else 0
        for classname in trainset.classes
    }

def mask_model(model, subnet):
    conv1_part = [n for n in subnet[0] if n < 10]
    conv2_part = [n - 10 for n in subnet[0] if 10 <= n < 30]
    fc1_part = [n - 30 for n in subnet[0] if n >= 30]

    if conv1_part:
        mask1 = torch.ones(model.conv1.weight.shape).to(device)
        for k in conv1_part:
            mask1[k] = 0
        prune.custom_from_mask(model.conv1, name='weight', mask=mask1)

    if conv2_part:
        mask2 = torch.ones(model.conv2.weight.shape).to(device)
        for k in conv2_part:
            mask2[k] = 0
        prune.custom_from_mask(model.conv2, name='weight', mask=mask2)

    if fc1_part:
        mask3 = torch.ones(model.fc1.weight.shape).to(device)
        for k in fc1_part:
            mask3[k] = 0
        prune.custom_from_mask(model.fc1, name='weight', mask=mask3)

def evaluate_subnets(ext):
    nets_path = os.path.join(OUTPUT_FOLDER, f"nets{ext}.p")
    subnets = [()] + load(open(nets_path, "rb"))

    R = {
        "Subnet": [], "Subnet_ID": [], "Overall_Accuracy": [],
        "F1": [], "Precision": [], "Recall": []
    }
    for class_id in IDX_TO_LABEL:
        R[IDX_TO_LABEL[class_id]] = []

    _, testloader = data_loaders(trainset, testset, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    for subnet_id, subnet in enumerate(subnets):
        print(f"\nTesting subnet {subnet_id}:")
        model = Net().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        torch.cuda.empty_cache()

        if subnet:
            if len(subnet[0]) > 2:
                mask_model(model, subnet)

        acc, f1, precision, recall = test_accuracy(model, testloader)
        class_acc = test_accuracy_per_class(model, testloader)

        R["Subnet"].append(subnet)
        R["Subnet_ID"].append(str(subnet_id))
        R["Overall_Accuracy"].append(acc.cpu().numpy())
        R["F1"].append(f1)
        R["Precision"].append(precision)
        R["Recall"].append(recall)

        print(f'Overall Accuracy: {(acc * 100):.2f} %')
        print('Per-class Accuracy:')
        for classname, acc_value in class_acc.items():
            print(f'{classname:12s}: {acc_value:.2f} %')
            R[classname].append(acc_value)

    df = pd.DataFrame(R)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df.to_excel(os.path.join(RESULTS_FOLDER, f"pruned{ext}.xlsx"))

def main():
    for ext in FILENAME_EXTENSIONS:
        print(f"\nProcessing {ext}")
        evaluate_subnets(ext)

if __name__ == "__main__":
    main()
