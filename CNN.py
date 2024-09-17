# -*- coding: utf-8 -*-
"""
@Time : 12/11/2023 5:05 PM
@Auth : Hao Yizhi
@File : CNN.py
@IDE  : PyCharm
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import logging
import time

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # Adjust this size to match the final conv output
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x / 255.0  # Normalize the image pixels to [0,1]
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Training function
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

            preds = outputs > 0.5
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1).long() == labels.data)

        # scheduler.step()  # Update the learning rate

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        logging.info(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


# Evaluation function
def evaluate_model(model, dataloaders, phase='val'):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = outputs.view(-1)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_pred.extend((probs > 0.5).long().cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    pr_auc = auc(tpr, fpr)

    # Logging
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info("Classification Report:\n" + clf_report)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))
    logging.info(f'ROC-AUC Score: {roc_auc:.4f}')
    logging.info(f'Precision-Recall AUC: {pr_auc:.4f}')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'cnn/{formatted_time} - CNN_ROC_curve.png')
    plt.close()


if __name__ == '__main__':
    # Set up logging
    local_time = time.localtime()
    formatted_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    logging.basicConfig(filename=f'cnn/{formatted_time} - CNN_classification.txt',
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Initialize the model, criterion, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Add a scheduler to decrease the learning rate over time
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    # Dataset and DataLoader setup
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder('../../Dataset/train', data_transforms['train']),
        'val': datasets.ImageFolder('../../Dataset/test', data_transforms['val']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=4),
        # shuffle=False for validation/test
    }

    # Train the model
    # model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
    model_trained = train_model(model, criterion, optimizer, num_epochs=25)

    # Evaluate the model
    evaluate_model(model_trained, dataloaders, phase='val')

    # Save the model
    torch.save(model_trained.state_dict(), f'cnn/{formatted_time} - CNN_model.pth')