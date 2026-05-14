
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# ---------------------- 全局配置（仅微调关键参数，其余不变） ----------------------
BATCH_SIZE = 64
LR = 1e-4  # 预训练模型用更小学习率，避免破坏预训练特征
FC_LR = 1e-3  # 顶层分类器学习率放大10倍
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data/"  # 你的images、train.csv、test.csv都放在这里
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
UNFREEZE_EPOCH = 10  # 第10轮解冻底层，分层微调


# ---------------------- 1. 数据集类（完全不变，保证对照） ----------------------
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


# ---------------------- 2. 数据增强（小幅增强，提升泛化，其余不变） ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------- 3. 加载数据（仅优化DataLoader，其余不变） ----------------------
# 训练集划分（完全不变）
train_df = pd.read_csv(TRAIN_CSV)
train_df_split, val_df_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['label']
)

# 保存临时CSV（完全不变）
train_df_split.to_csv(os.path.join(DATA_ROOT, "train_split.csv"), index=False)
val_df_split.to_csv(os.path.join(DATA_ROOT, "val_split.csv"), index=False)

# 构建数据集（完全不变）
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

# DataLoader（优化：pin_memory=True加速GPU传输，其余不变）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


# ---------------------- 4. 模型替换：ResNeXt50 微调（核心改动） ----------------------
def get_resnext50(num_classes):
    # 加载预训练ResNeXt50（ImageNet权重）
    model = models.resnext50_32x4d(pretrained=True)

    # 初始冻结所有卷积层，只训练顶层（提速+避免过拟合）
    for param in model.parameters():
        param.requires_grad = False

    # 替换顶层分类器，适配树叶分类任务
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # 防过拟合
        nn.Linear(in_features, num_classes)
    )

    # 移到指定设备
    return model.to(DEVICE)


# ---------------------- 5. 训练/验证函数（完全不变，保证对照） ----------------------
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


# ---------------------- 6. 测试集预测（完全不变） ----------------------
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
    test_df[["image", "label"]].to_csv("submission_resnext.csv", index=False)
    print("✅ ResNeXt 提交文件已生成：submission_resnext.csv")


# ---------------------- 绘图函数（完全不变） ----------------------
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path="resnext_curve.png"):
    # 1. 设置字体（解决中文显示问题，同时兼容英文）
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]  # 优先黑体，后备英文
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 2. 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ResNeXt50 Training Curves (Leaf Classification)", fontsize=14, fontweight="bold")

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


# ---------------------- 7. 主流程（仅优化优化器/调度器，其余不变） ----------------------
if __name__ == "__main__":
    num_classes = train_dataset.num_classes
    print(f"📊 数据集类别数：{num_classes} | 设备：{DEVICE}")

    # 初始化模型：替换为ResNeXt50
    model = get_resnext50(num_classes)
    criterion = nn.CrossEntropyLoss()

    # 优化器：分层学习率（顶层大学习率，底层小学习率）
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': FC_LR},  # 顶层分类器
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': LR}  # 卷积层
    ], weight_decay=2e-4)

    # 学习率调度器：余弦退火重启，更适配微调
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 训练记录（完全不变）
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # 开始训练（新增解冻逻辑，其余不变）
    print("\n🚀 开始训练 ResNeXt50...")
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        # 第10轮解冻最后两层卷积，精细微调
        if epoch == UNFREEZE_EPOCH:
            print("🔓 解冻ResNeXt50最后两层卷积层，精细微调...")
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "resnext_best.pth")
            print(f"💾 最优模型更新 | 验证准确率：{best_val_acc:.2f}%")

        print(f"📈 训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.2f}%")
        print(f"📉 验证损失：{val_loss:.4f} | 验证准确率：{val_acc:.2f}%")

    print(f"\n🎉 ResNeXt50 训练完成 | 最佳验证准确率：{best_val_acc:.2f}%")

    # 绘图（仅改保存名，其余不变）
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    print("📊 训练曲线已保存：resnext_curve.png")

    # 测试集预测（仅改模型加载名，其余不变）
    print("\n🔍 开始测试集预测...")
    model.load_state_dict(torch.load("resnext_best.pth"))
    predict_test(model, test_loader, train_dataset.label_map, DEVICE)

