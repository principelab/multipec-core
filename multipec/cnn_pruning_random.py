import os
from pickle import load
import pandas as pd
from random import sample, choice, seed
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

# Set random seed
torch.backends.cudnn.enabled = False
torch.manual_seed(SEED)
seed(42)

transform = T.Compose([T.ToTensor(),
                                 T.Normalize((0.1307,), (0.3081,))])

trainset = MNIST('/files/', train=True, download=True, transform=transform)

testset = MNIST('/files/', train=False, download=True, transform=transform)

IDX_TO_LABEL = {v: k for k, v in testset.class_to_idx.items()}
n_classes = len(IDX_TO_LABEL)

# number of subprocesses to use for data loading
NUM_WORKERS = 2

def data_loaders(trainset, testset, trainsize, testsize):
    trainloader = DataLoader(
    trainset, batch_size=trainsize, 
    num_workers=NUM_WORKERS
    )
    testloader = DataLoader(
        testset, batch_size=testsize, 
        num_workers=NUM_WORKERS, shuffle=True
    )
    return trainloader, testloader

def test_accuracy(model, testloader):
    correct = 0
    f1_list, precision_list, recall_list = [],[],[]
    labels_sum, output_sum = [], []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the modelwork
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            predicted = outputs.data.max(1, keepdim=True)[1]

            correct += predicted.eq(labels.data.view_as(predicted)).sum()

            labels_sum += labels
            output_sum += predicted

    accuracy = correct / len(testloader.dataset)
    f1 = f1_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)
    precision = precision_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)
    recall = recall_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)

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

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[trainset.classes[label]] += 1
                total_pred[trainset.classes[label]] += 1
    
    accuracy_per_class = {classname: 0 for classname in trainset.classes}
    for classname, correct_count in correct_pred.items():
        accuracy = (100 * float(correct_count)) / total_pred[classname]
        accuracy_per_class[classname] = accuracy

    return accuracy_per_class

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
    
R = {"Subnet":[], "Subnet_ID":[], "Overall_Accuracy":[], "F1":[], "Precision":[], "Recall":[]}
for class_id in IDX_TO_LABEL:
    R[IDX_TO_LABEL[class_id]]=[]

load_subnets = load(open(OUTPUT_FOLDER+f"nets.p", "rb"))

min_group_size = min([len(x[0]) for x in load_subnets])
max_group_size = max([len(x[0]) for x in load_subnets])

size_options = list(range(min_group_size,max_group_size,1))
print(size_options)

nodes = list(range(30))

rd_runs = 100
rd_subnets = [sample(nodes, choice(size_options)) for i in range(rd_runs)]
rd_subnets = [()]+rd_subnets

print(f"Net: MNIST model (2 layers)")
print("Number of subnets:", len(rd_subnets))

subnet_id = 0
for subnet in rd_subnets:
    print(subnet)
    if not subnet:

        R["Subnet"].append(subnet)
        R["Subnet_ID"].append(str(subnet_id))
        print(f"Testing subnet {subnet_id}:")
        subnet_id+=1

        torch.cuda.empty_cache()

        _, testloader = data_loaders(
            trainset, testset, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

        # Load model
        model = Net()
        model.to(device)
        model.load_state_dict(torch.load(MODEL_PATH))

        overall_accuracy, f1, precision, recall = test_accuracy(model, testloader)
        R["Overall_Accuracy"].append(overall_accuracy.cpu().numpy())
        R["F1"].append(f1)
        R["Precision"].append(precision)
        R["Recall"].append(recall)

        print('Overall accuracy of the network  '
            f'{(overall_accuracy * 100):.2f} %\n'
            'on the 1000 test images')

        accuracy_per_class = test_accuracy_per_class(model, testloader)

        print('Accuracy per class\n')
        for classname, accuracy in accuracy_per_class.items():
            print(f'{classname:12s} {accuracy:.2f} %')
            R[classname].append(accuracy)      

    elif len(subnet)>2:

        R["Subnet"].append(subnet)
        R["Subnet_ID"].append(str(subnet_id))
        print(f"Testing subnet {subnet_id}:")
        subnet_id+=1

        torch.cuda.empty_cache()

        _, testloader = data_loaders(
            trainset, testset, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

        # Load model
        model = Net()
        model.to(device)
        model.load_state_dict(torch.load(MODEL_PATH))

        conv1_part = [node for node in subnet if node<10]
        conv2_part = [node-10 for node in subnet if node>=10]

        mask1 = torch.ones(model.conv1.weight.shape).to(device)
        for kernel in conv1_part:
            mask1[kernel] = torch.zeros(model.conv1.weight.shape[1::]).to(device)
        prune.custom_from_mask(module=model.conv1, name='weight', mask=mask1) # zero the weights (not the biases)

        mask2 = torch.ones(model.conv2.weight.shape).to(device)
        for kernel in conv2_part:
            mask2[kernel] = torch.zeros(model.conv2.weight.shape[1::]).to(device)
        prune.custom_from_mask(module=model.conv2, name='weight', mask=mask2) # zero the weights (not the biases)

        overall_accuracy, f1, precision, recall = test_accuracy(model, testloader)

        R["Overall_Accuracy"].append(overall_accuracy.cpu().numpy())
        R["F1"].append(f1)
        R["Precision"].append(precision)
        R["Recall"].append(recall)

        print('Overall accuracy of the network  '
            f'{(overall_accuracy * 100):.2f} %\n'
            'on the 1000 test images')

        accuracy_per_class = test_accuracy_per_class(model, testloader)

        print('Accuracy per class\n')
        for classname, accuracy in accuracy_per_class.items():
            print(f'{classname:12s} {accuracy:.2f} %')
            R[classname].append(accuracy)

    df = pd.DataFrame(R)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df.to_excel(os.path.join(RESULTS_FOLDER, f"pruned_random_{rd_runs}runs.xlsx"))
