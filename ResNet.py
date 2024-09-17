# -*- coding: utf-8 -*-
"""
@Time : 11/4/2023 11:00 PM
@Auth : Wang Yuyang
@File : ResNet.py
@IDE  : PyCharm
"""
import torch
import csv
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, \
    precision_recall_fscore_support

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = '../../Dataset/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=0, pin_memory=True)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will run on {device}.")

batch_counter = 0


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, result_file_name="defalut.csv"):
    with open("res/" + result_file_name, 'w', newline="") as csvfile:
        csvfile.truncate()
        writer = csv.writer(csvfile)
        writer.writerow(["epoch_" + result_file_name, "train_loss", "train_acc"])

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    tqdm_iter = tqdm(range(num_epochs))
    for epoch in tqdm_iter:
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            curr_batch_count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    global batch_counter
                    batch_counter += 1
                    with open("res/" + result_file_name, "a", newline="") as file:
                        csv_file = csv.writer(file, delimiter=',')
                        csv_file.writerow([
                            batch_counter,
                            loss.item() * inputs.size(0),
                            float(torch.sum(preds == labels.data) / inputs.size(0))
                        ])
                curr_batch_count += 1
                tqdm_iter.set_description(
                    f"Batch: {curr_batch_count}/{len(dataloaders[phase])}, Batch Loss: {loss.item() * inputs.size(0)}, Acc: {torch.sum(preds == labels.data) / inputs.size(0)}")
                # print("Batch Loss: {}, Acc: {}".format(loss.item() * inputs.size(0),
                #                                        torch.sum(preds == labels.data) / inputs.size(0)))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'res/best_model_wts.pth')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                print(images_so_far)
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.axis('off')

                ax.set_title('real: {}'
                             '\npredicted: {}'.format(class_names[labels[j]],
                                                      class_names[preds[j]]
                                                      ))

                # print(torch.mean(inputs.cpu().data[j]).shape)

                plt.imshow(torch.permute(inputs.cpu().data[j], (1, 2, 0)))
                # Adjust padding after creating all subplots
                plt.subplots_adjust(hspace=1)  # the amount of height reserved for space between subplots,
                # expressed as a fraction of the average axis height
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    # plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloaders, phase='test'):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders[phase]):
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
    print(f'Accuracy: {accuracy:.4f}')
    print("Classification Report:\n" + clf_report)
    print("Confusion Matrix:\n" + str(conf_matrix))
    print(f'ROC-AUC Score: {roc_auc:.4f}')
    print(f'Precision-Recall AUC: {pr_auc:.4f}')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig(f'cnn/{formatted_time} - CNN_ROC_curve.png')
    plt.close()

model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
#
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Get the current local time
local_time = time.localtime()

# Format the time in a specific format
formatted_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20, result_file_name=f"{formatted_time} - resnet18_LR_0.00001.csv")

# plt.ioff()
# plt.show()

from sklearn.metrics import roc_curve, auc
from itertools import cycle

model_ft.load_state_dict(torch.load('res/best_model_wts.pth'))
# visualize_model(model_ft)
# evaluate_model(model_ft, dataloaders)
def plot_roc_curve(model, num_classes):
    model.eval()
    y_true = []
    y_scores = []
    count = [0]
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Convert labels to one-hot encoding
            y_true.append(labels.cpu().numpy())
            # Append the scores from the outputs
            y_scores.append(outputs.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            # if count[0] == 10:
            #      break
            # else:
            #     count[0] = count[0] + 1

    # Concatenate all the batches
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    y_pred = np.concatenate(y_pred)
    # One-hot encode the labels
    y_true = label_binarize(y_true, classes=range(num_classes))

    # Compute ROC curve and ROC area for each class
    print()
    print(y_true.shape, y_scores.shape)
    fpr, tpr, _ = roc_curve([i[0] for i in y_true], [i[1] for i in y_scores])
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # Plot all ROC curves
    # Plot ROC curve
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(roc_auc),
             linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curve')
    plt.legend(loc="lower right")

    # Plot confusion matrix
    plt.subplot(1, 3, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Labeling the plot
    thresh = cm.max() / 2.  # threshold for text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # Plot precision, recall, and F1 score
    plt.subplot(1, 3, 3)
    metrics = [precision, recall, f1_score]
    metric_names = ['Precision', 'Recall', 'F1 Score']
    # plt.bar(metric_names, metrics)
    plt.ylim(0, 1)
    bars = plt.bar(metric_names, metrics)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.1, round(yval, 2), ha='center', va='bottom', color='white')

    plt.title('Precision, Recall, F1 Score')

    plt.tight_layout()

    # Print accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')

    plt.show()


plot_roc_curve(model_ft, len(class_names))
