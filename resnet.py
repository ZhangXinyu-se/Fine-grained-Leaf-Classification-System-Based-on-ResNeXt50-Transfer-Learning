# *********自定义ResNet

# 神经网络架构：
# 自定义ResNet：实现了残差网络结构
# 残差块（ResidualBlock）：包含跳跃连接（skip connection）
# 批归一化（BatchNorm2d）：加速训练、稳定模型
# 自适应平均池化（AdaptiveAvgPool2d）：将任意尺寸的特征图转换为固定大小
# ReLU激活函数：引入非线性
# 训练技术：
# AdamW优化器：Adam的改进版，带有解耦的权重衰减
# 余弦退火学习率调度：动态调整学习率
# 交叉熵损失：多分类任务的损失函数
# 权重衰减：L2正则化防止过拟合
#  数据处理
# 自定义Dataset类：继承PyTorch的Dataset
# DataLoader：批量加载数据
# 数据增强：
# 随机水平翻转
# 随机旋转
# 归一化（ImageNet统计量）
# PIL图像处理：读取和处理图像

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# https://www.kaggle.com/competitions/classify-leaves/data?select=images
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------- 全局配置 ----------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data/"  # 你的images、train.csv、test.csv都放在这里
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")


# ---------------------- 1. 数据集类 ----------------------
class LeafDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            # 训练集：标签编码
            self.label_map = {v: i for i, v in enumerate(sorted(self.df['label'].unique()))}
            self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        full_img_path = os.path.join(self.data_root, img_path)
        img = Image.open(full_img_path).convert("RGB")  # 彩色图

        if self.transform:
            img = self.transform(img)

        if self.is_train:
            label = self.df.iloc[idx]['label']
            label = self.label_map[label]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img


# ---------------------- 2. 数据增强 ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    #随机裁剪，提升泛化（仅加这一行）
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #中心裁剪，避免边缘噪声（仅加这一行）
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------- 3. 加载数据 ----------------------
# 训练集划分
train_df = pd.read_csv(TRAIN_CSV)
train_df_split, val_df_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['label']
)

# 保存临时CSV，方便数据集读取
train_df_split.to_csv(os.path.join(DATA_ROOT, "train_split.csv"), index=False)
val_df_split.to_csv(os.path.join(DATA_ROOT, "val_split.csv"), index=False)

# 构建数据集
train_dataset = LeafDataset(
    csv_path=os.path.join(DATA_ROOT, "train_split.csv"),
    data_root=DATA_ROOT,
    transform=train_transform,
    is_train=True
)
val_dataset = LeafDataset(
    csv_path=os.path.join(DATA_ROOT, "val_split.csv"),
    data_root=DATA_ROOT,
    transform=val_test_transform,
    is_train=True
)
test_dataset = LeafDataset(
    csv_path=TEST_CSV,
    data_root=DATA_ROOT,
    transform=val_test_transform,
    is_train=False
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ---------------------- 4. ResNet 模型 ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(32, 32, stride=2)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 128, stride=2)
        self.res4 = ResidualBlock(128, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


# ---------------------- 5. 训练/验证函数 ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


# ---------------------- 6. 测试集预测 ----------------------
def predict_test(model, test_loader, label_map, device):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Predicting Test Set"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            test_preds.extend(pred.cpu().numpy())

    # 标签反编码
    inv_label_map = {v: k for k, v in label_map.items()}
    test_df = pd.read_csv(TEST_CSV)
    test_df["label"] = [inv_label_map[p] for p in test_preds]
    test_df[["image", "label"]].to_csv("submission_resnet.csv", index=False)
    print("✅ ResNet 提交文件已生成：submission_resnet.csv")

# ---------------------- 新增：修复后的绘图函数 ----------------------
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path="resnet_curve.png"):
    # 1. 设置字体（解决中文显示问题，同时兼容英文）
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]  # 优先黑体，后备英文
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 2. 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ResNet Training Curves (Leaf Classification)", fontsize=14, fontweight="bold")

    # 3. 绘制准确率曲线（左图）
    ax1.plot(train_accs, label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(val_accs, label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs. Epoch", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)  # 准确率固定0-100%

    # 4. 绘制损失曲线（右图）
    ax2.plot(train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss Value", fontsize=12)
    ax2.set_title("Loss vs. Epoch", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 5. 保存并显示
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------- 7. 主流程 ----------------------
if __name__ == "__main__":
    num_classes = train_dataset.num_classes
    print(f"📊 数据集类别数：{num_classes} | 设备：{DEVICE}")

    # 初始化模型
    model = ResNet(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练记录
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # 开始训练
    print("\n🚀 开始训练 ResNet...")
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "resnet_best.pth")
            print(f"💾 最优模型更新 | 验证准确率：{best_val_acc:.2f}%")

        print(f"📈 训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.2f}%")
        print(f"📉 验证损失：{val_loss:.4f} | 验证准确率：{val_acc:.2f}%")

    print(f"\n🎉 ResNet 训练完成 | 最佳验证准确率：{best_val_acc:.2f}%")

    # ---------------------- 替换：调用新的绘图函数 ----------------------
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    print("📊 训练曲线已保存：resnet_curve.png")

    # 测试集预测
    print("\n🔍 开始测试集预测...")
    model.load_state_dict(torch.load("resnet_best.pth"))
    predict_test(model, test_loader, train_dataset.label_map, DEVICE)