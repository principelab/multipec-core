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
    

# Load unpruned model once and get baseline class accuracy
_, testloader = data_loaders(trainset, testset, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)
model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
baseline_class_accuracies = test_accuracy_per_class(model, testloader)

# Class specificity (Top2)
def compute_top2_class_specificity(deltas: dict, class_name: str):
    others = [abs(v) for k, v in deltas.items() if k != class_name]
    if len(others) < 2:
        return 0.0
    top2 = sorted(others, reverse=True)[:2]
    return top2[0] / (top2[1] + 1e-8)

def compute_maxmean_class_specificity(deltas: dict, class_name: str):
    others = [abs(v) for k, v in deltas.items() if k != class_name]
    if not others:
        return 0.0
    max_val = max(others)
    mean_val = sum(others) / len(others)
    return max_val / (mean_val + 1e-8)

# MultiPEC results
class_best_top2_map = {
    "1 - one": 4.479203,
    "2 - two": 1.739879,
    "3 - three": 1.753375,
    "4 - four": 3.085540,
    "5 - five": 1.777546,
    "7 - seven": 1.705447,
    "8 - eight": 3.290615,
    "9 - nine": 3.753992
}
class_best_maxmean_map = {
    "1 - one": 5.185341,
    "2 - two": 3.364931,
    "3 - three": 3.069058,
    "4 - four": 4.816503,
    "5 - five": 3.085191,
    "7 - seven": 2.934055,
    "8 - eight": 5.953951,
    "9 - nine": 4.826446
}

# Track which classes have matched their target specificity
class_success_flags = {cls: False for cls in class_best_top2_map}
class_success_iters = {cls: None for cls in class_best_top2_map}

# Track all iterations
MAX_GLOBAL_ITERATIONS = 100
detailed_results = []

# Define random networks
load_subnets = load(open(OUTPUT_FOLDER+f"nets.p", "rb"))

min_group_size = min([len(x[0]) for x in load_subnets])
max_group_size = max([len(x[0]) for x in load_subnets])

size_options = list(range(min_group_size,max_group_size,1))
print(size_options)

nodes = list(range(30))

rd_runs = MAX_GLOBAL_ITERATIONS
rd_subnets = [sample(nodes, choice(size_options)) for i in range(rd_runs)]
rd_subnets = [()]+rd_subnets

print(f"Net: MNIST model (2 layers)")
print("Number of subnets:", len(rd_subnets))

for iteration_idx, subnet in enumerate(rd_subnets):
    if iteration_idx >= MAX_GLOBAL_ITERATIONS:
        print("Reached global iteration limit.")
        break
    if all(class_success_flags.values()):
        print("All class specificities reached!")
        break

    print(f"\n=== Iteration {iteration_idx + 1} ===")
    torch.cuda.empty_cache()
    _, testloader = data_loaders(trainset, testset, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    # Load and optionally prune model
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    if subnet:
        conv1_part = [node for node in subnet if node < 10]
        conv2_part = [node - 10 for node in subnet if node >= 10]

        mask1 = torch.ones(model.conv1.weight.shape).to(device)
        for kernel in conv1_part:
            mask1[kernel] = 0
        prune.custom_from_mask(model.conv1, 'weight', mask1)

        mask2 = torch.ones(model.conv2.weight.shape).to(device)
        for kernel in conv2_part:
            mask2[kernel] = 0
        prune.custom_from_mask(model.conv2, 'weight', mask2)

    # Evaluate
    overall_accuracy, f1, precision, recall = test_accuracy(model, testloader)
    pruned_class_accuracies = test_accuracy_per_class(model, testloader)

    # Compute Δs for all classes
    deltas = {
        cname: pruned_class_accuracies.get(cname, 0) - baseline_class_accuracies.get(cname, 0)
        for cname in IDX_TO_LABEL.values()
    }

    # Compute specificity for all classes
    specificity_per_class = {
        cname: compute_maxmean_class_specificity(deltas, cname)
        for cname in class_best_top2_map
    }

    # Identify most specific class this subnet targets
    specific_class = max(specificity_per_class, key=specificity_per_class.get)
    spec_value = specificity_per_class[specific_class]
    target_spec = class_best_maxmean_map[specific_class]
    is_success = spec_value >= target_spec and not class_success_flags[specific_class]

    # Mark class as matched if applicable
    if is_success:
        class_success_flags[specific_class] = True
        class_success_iters[specific_class] = iteration_idx + 1
        print(f"Matched class specificity for '{specific_class}' at iteration {iteration_idx + 1} ({spec_value:.4f})")

    # Record the iteration result
    detailed_results.append({
        "Iteration": iteration_idx + 1,
        "Subnet": subnet,
        "Specific_Class": specific_class,
        "Specificity": spec_value,
        "Target_Specificity": target_spec,
        "Successful": is_success
    })
 

 # Save full iteration log
detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_excel(os.path.join(RESULTS_FOLDER, "global_random_pruning_log.xlsx"), index=False)

# Summary
summary_df = pd.DataFrame([
    {"Class": cls, "Reached": success, "Iteration": class_success_iters[cls]}
    for cls, success in class_success_flags.items()
])
summary_df.to_excel(os.path.join(RESULTS_FOLDER, "class_specificity_summary.xlsx"), index=False)

print("\nSummary:")
print(summary_df)


