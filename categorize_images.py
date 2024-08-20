!pip install torch torchvision
!pip install tqdm
!pip install torchviz
!pip install torchsummary

import os
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torch.nn.functional as F
from torchviz import make_dot
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ拡張と前処理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
])

transform_val_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root='yourpath_train', transform=transform_train)
val_dataset = ImageFolder(root='yourpath_val', transform=transform_val_test)
test_dataset = ImageFolder(root='yourpath_test', transform=transform_val_test)

num_classes = len(train_dataset.classes)

# クラスごとの重みを計算
class_counts = [0] * num_classes
for _, label in train_dataset:
    class_counts[label] += 1
class_weights = torch.tensor([max(class_counts) / count for count in class_counts]).to(device)

best_batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=1):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1)
        )
        self.dropout_prob = dropout_prob

    def forward(self, x):
        # ドロップアウトを適用する確率を決定
        if self.training and self.dropout_prob > 0:
            apply_dropout = torch.rand(1).item() < self.dropout_prob
        else:
            apply_dropout = False

        if apply_dropout:
            # ドロップアウトを適用する場合、入力xをそのまま返す
            return x
        else:
            # 自己注意機構の計算
            batch_size, seq_len, feature_dim = x.size()
            x = x.view(batch_size * seq_len, feature_dim)
            weights = self.attention(x)
            weights = weights.view(batch_size, seq_len)
            weights = torch.softmax(weights, dim=1)
            weights = weights.unsqueeze(-1)
            weighted = x.view(batch_size, seq_len, feature_dim) * weights
            return weighted.sum(dim=1)

class AttentionResNet(nn.Module):
    def __init__(self, num_classes):
        super(AttentionResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.self_attention = Attention(input_dim=512, output_dim=512)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = x.unsqueeze(1)  # シーケンス次元を追加
        x = self.self_attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = AttentionResNet(num_classes).to(device)

# トレーニングと評価の設定
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

num_epochs = 20

train_losses = []
val_losses = []
val_accuracies = []
val_mcc_scores = []
val_f1_scores = []
val_sensitivities = []
val_specificities = []
val_roc_aucs = []

test_accuracies = []
test_mcc_scores = []
test_f1_scores = []
test_sensitivities = []
test_specificities = []
test_roc_aucs = []

# 交差検証の定義
k_folds = 5
splits = list(train_test_split(np.arange(len(train_dataset)), test_size=1/k_folds))

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(outputs.cpu().tolist())

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_mcc = matthews_corrcoef(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_sensitivity = sum((pred == true == 1) for pred, true in zip(all_preds, all_labels)) / sum(true == 1 for true in all_labels)
    val_specificity = sum((pred == true == 0) for pred, true in zip(all_preds, all_labels)) / sum(true == 0 for true in all_labels)
    val_roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_mcc_scores.append(val_mcc)
    val_f1_scores.append(val_f1)
    val_sensitivities.append(val_sensitivity)
    val_specificities.append(val_specificity)
    val_roc_aucs.append(val_roc_auc)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation MCC: {val_mcc:.4f}, Validation F1: {val_f1:.4f}, Validation Sensitivity: {val_sensitivity:.4f}, Validation Specificity: {val_specificity:.4f}, Validation ROC-AUC: {val_roc_auc:.4f}')

    scheduler.step(val_loss)

    # テストデータでの評価
    model.eval()
    correct_test = 0
    total_test = 0
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_test_preds.extend(predicted.cpu().tolist())
            all_test_labels.extend(labels.cpu().tolist())
            all_test_probs.extend(outputs.cpu().tolist())

    test_accuracy = correct_test / total_test
    test_mcc = matthews_corrcoef(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_sensitivity = sum((pred == true == 1) for pred, true in zip(all_test_preds, all_test_labels)) / sum(true == 1 for true in all_test_labels)
    test_specificity = sum((pred == true == 0) for pred, true in zip(all_test_preds, all_test_labels)) / sum(true == 0 for true in all_test_labels)
    test_roc_auc = roc_auc_score(all_test_labels, [prob[1] for prob in all_test_probs])
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    tn, fp, fn, tp = cm.ravel()

    test_accuracies.append(test_accuracy)
    test_mcc_scores.append(test_mcc)
    test_f1_scores.append(test_f1)
    test_sensitivities.append(test_sensitivity)
    test_specificities.append(test_specificity)
    test_roc_aucs.append(test_roc_auc)

    print(f'Test Accuracy: {test_accuracy:.4f}, Test MCC: {test_mcc:.4f}, Test F1: {test_f1:.4f}, Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}, Test ROC-AUC: {test_roc_auc:.4f}')
    print(f'True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}')

# テストデータでの最終評価
model.eval()
correct_test = 0
total_test = 0
all_test_preds = []
all_test_labels = []
all_test_probs = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Final Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_test_preds.extend(predicted.cpu().tolist())
        all_test_labels.extend(labels.cpu().tolist())
        all_test_probs.extend(outputs.cpu().tolist())

final_test_accuracy = correct_test / total_test
final_test_mcc = matthews_corrcoef(all_test_labels, all_test_preds)
final_test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
final_test_sensitivity = sum((pred == true == 1) for pred, true in zip(all_test_preds, all_test_labels)) / sum(true == 1 for true in all_test_labels)
final_test_specificity = sum((pred == true == 0) for pred, true in zip(all_test_preds, all_test_labels)) / sum(true == 0 for true in all_test_labels)
final_test_roc_auc = roc_auc_score(all_test_labels, [prob[1] for prob in all_test_probs])

print(f'Final Test Accuracy: {final_test_accuracy:.4f}, Final Test MCC: {final_test_mcc:.4f}, Final Test F1: {final_test_f1:.4f}, Final Test Sensitivity: {final_test_sensitivity:.4f}, Final Test Specificity: {final_test_specificity:.4f}, Final Test ROC-AUC: {final_test_roc_auc:.4f}')