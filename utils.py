import numpy as np
import torch.nn.functional as F
import torch
import torch.nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def evaluate(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def visualize_dataset(dataset, num_images=3):
  fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
  class_indices = [0, dataset.__len__() // 3, (dataset.__len__() * 2) // 3]
  
  for i, class_index in enumerate(class_indices):
    image, label = dataset[class_index]

    image = image.numpy().transpose((1, 2, 0))

    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label], fontsize=10, wrap=True)
    axes[i].axis('off')
  plt.tight_layout()
  plt.savefig("Lung Cancers")
  plt.show()
    
    