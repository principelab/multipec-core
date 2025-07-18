import os
import pandas as pd
from random import randrange
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import models

from sklearn.metrics import f1_score, precision_score, recall_score

SEED = 1
torch.manual_seed(SEED)

input_folder = "data/input/cnn/"
path_model = input_folder + 'model.pth'
path_optimizer = input_folder + 'optimizer.pth'

output_folder = "data/output/cnn/"
os.makedirs(output_folder, exist_ok=True)

os.makedirs(output_folder+'model-states/outputs', exist_ok=True)
os.makedirs(output_folder+'model-states/weights', exist_ok=True)
os.makedirs(output_folder+'model-states/train_scores', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_epochs = 5

batch_size_task = 64
N_outputs_per_class = 20
N = N_outputs_per_class * batch_size_task

batch_size_test = 1000

learning_rate = 0.01
momentum = 0.5
log_interval = 10

transform = transform=T.Compose([T.ToTensor(),
                                 T.Normalize((0.1307,), (0.3081,))])

trainset = MNIST('/files/', train=True, download=True, transform=transform)
testset = MNIST('/files/', train=False, download=True, transform=transform)

IDX_TO_LABEL = {v: k for k, v in testset.class_to_idx.items()}
n_classes = len(IDX_TO_LABEL)

def data_loaders(taskset, testset, tasksize, testsize):
    taskloader = torch.utils.data.DataLoader(
    taskset, batch_size=tasksize,
    num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testsize,
        num_workers=2
    )
    return taskloader, testloader

task_dataloader, _ = data_loaders(trainset, testset, tasksize=1, testsize=batch_size_test)

image_per_class_counter = {k:0 for k in IDX_TO_LABEL}

images_per_class = {k:[] for k in IDX_TO_LABEL}

for batch_i, (inputs, labels) in enumerate(task_dataloader):

    for l in labels:
      image_per_class_counter[int(l)] += 1

    if all(i > N for i in list(image_per_class_counter.values())): break

    else:
      if image_per_class_counter[int(labels[0])] > N: pass
      else:
        images_per_class[int(l)].append(inputs[0])

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

model = Net()
model.load_state_dict(torch.load(path_model))

from random import shuffle

def task(model, epochs=5, N_img_per_class=20):

    # Drop the Dropoutout layers
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    # Add hooks to achieve outputs extraction
    outputs_dict = {}
    def get_features(name):
        def hook(model, input, output):
            outputs_dict[name] = output[0].detach()
        return hook

    counter = 1
    for layer in model.modules():
        name = layer._get_name()
        layer.register_forward_hook(get_features(str(counter)+'_'+name))
        counter += 1

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.load_state_dict(torch.load(path_optimizer))

    task_loss_list, test_loss_list = [],[]
    accuracy_list = []
    f1_list, precision_list, recall_list = [],[],[]

    labels_sum, output_sum = [], []
    
    for epoch in range(epochs):

        _, testloader = data_loaders(trainset, testset, tasksize=batch_size_task, testsize=batch_size_test)

        INPUTS, LABELS = [],[]
        n=0
        while n<N_img_per_class:
          for label in images_per_class:
            INPUTS+=images_per_class[label][n:n+batch_size_task]
            LABELS+=[label]*batch_size_task
          n+=1

        taskloader = DataLoader([[INPUTS[i], LABELS[i]] for i in range(len(LABELS))], shuffle=True, batch_size=batch_size_task)

        model.train()
        for batch_i, (inputs, labels) in enumerate(taskloader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            pred = output.data.max(1, keepdim=True)[1]
            correct_prediction = pred.eq(labels.data.view_as(pred)).sum()

            loss = F.nll_loss(output, labels)

            loss.backward()
            optimizer.step()

            task_loss_list.append(loss.item())

            # Save parameters and output
            state = model.state_dict()
            batch_class = labels[0]
            torch.save(state, output_folder+f'model-states/weights/E{epoch}_B{batch_i}_{batch_class}_{correct_prediction}_weights.pth')
            torch.save(outputs_dict, output_folder+f'model-states/outputs/E{epoch}_B{batch_i}_{batch_class}_{correct_prediction}_outputs.pth')

        correct = 0
        test_loss = 0.0
        model.eval()
        for inputs, labels in testloader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                output = model(inputs)
                loss = F.nll_loss(output, labels, reduction='sum')

                test_loss += loss.item()

                predicted = output.data.max(1, keepdim=True)[1]

                correct += predicted.eq(labels.data.view_as(predicted)).sum()

                labels_sum += labels
                output_sum += predicted

        test_loss /= len(testloader.dataset)
        accuracy = (correct / len(testloader.dataset))

        test_loss_list.append(test_loss)
        accuracy_list.append(accuracy.cpu().detach().numpy())

        f1 = f1_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)
        precision = precision_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)
        recall = recall_score(torch.Tensor(labels_sum).to("cpu"), torch.Tensor(output_sum).to("cpu"), average=None, zero_division=0)

        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

    return task_loss_list, test_loss_list, accuracy_list, f1_list, precision_list, recall_list

# Task execution
train_loss_list, test_loss_list, accuracy_list, f1_list, precision_list, recall_list = task(model, epochs=n_epochs, N_img_per_class=20)

# Save metrics to control the task performance 
metrics_df = pd.DataFrame({
    "Epoch": list(range(1, len(train_loss_list) + 1)),
    "Train Loss": train_loss_list,
    "Test Loss": test_loss_list,
    "Accuracy": accuracy_list,
    "F1 Score": f1_list,
    "Precision": precision_list,
    "Recall": recall_list
})

excel_path = os.path.join(output_folder, 'model-states', 'train_scores', 'task_metrics.xlsx')
os.makedirs(os.path.dirname(excel_path), exist_ok=True)
metrics_df.to_excel(excel_path, index=False)
