import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import time
import os

# === CONFIG ===
DATA_DIR = "/research/duwe/Sri/ILSVRC"  # <-- UPDATE this
NUM_CLASSES = 1000
BATCH_SIZE = 256
EPOCHS = 10
ENTROPY_THRESHOLDS = [0.1, 0.2, 0.3]
CHECKPOINT_PATH = "./branchy_resnet50_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# === TRANSFORMS ===
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === DATA LOADERS ===
num_workers = os.cpu_count() // 2

train_loader = torch.utils.data.DataLoader(
    ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
)

# === BRANCHY RESNET50 ===
class BranchyResNet50(nn.Module):
    def __init__(self, thresholds):
        super().__init__()
        base = torchvision.models.resnet50(weights=None)
        self.thresholds = thresholds

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = base.avgpool
        self.final_fc = base.fc

        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, NUM_CLASSES))
        self.exit2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, NUM_CLASSES))
        self.exit3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, NUM_CLASSES))

    def forward(self, x):
        outputs = []
        entropies = []

        x = self.stem(x)
        x1 = self.layer1(x)
        out1 = self.exit1(x1)
        outputs.append(out1)
        entropies.append(self._entropy(out1))

        x2 = self.layer2(x1)
        out2 = self.exit2(x2)
        outputs.append(out2)
        entropies.append(self._entropy(out2))

        x3 = self.layer3(x2)
        out3 = self.exit3(x3)
        outputs.append(out3)
        entropies.append(self._entropy(out3))

        x4 = self.layer4(x3)
        x4 = self.pool(x4).view(x4.size(0), -1)
        out4 = self.final_fc(x4)
        outputs.append(out4)

        final_output = out4.clone()
        exit_points = torch.full((x.shape[0],), 4, dtype=torch.int, device=x.device)
        for i in range(3):
            confident = entropies[i] < self.thresholds[i]
            final_output[confident] = outputs[i][confident]
            exit_points[confident] = i + 1
        return final_output, exit_points, outputs

    def _entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

# === TRAINING LOOP ===
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, exits = model(imgs)
            loss = (
                0.2 * criterion(exits[0], labels) +
                0.2 * criterion(exits[1], labels) +
                0.2 * criterion(exits[2], labels) +
                0.4 * criterion(exits[3], labels)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss

# === EVALUATION LOOP ===
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    exit_counts = [0, 0, 0, 0]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds, exits, _ = model(imgs)
            predicted = preds.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for i in range(1, 5):
                exit_counts[i-1] += (exits == i).sum().item()
    acc = 100 * correct / total
    return acc, exit_counts

# === MAIN ===
if __name__ == "__main__":
    model = BranchyResNet50(thresholds=ENTROPY_THRESHOLDS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        loss = train(model, train_loader, optimizer, criterion, scaler)
        acc, exits = evaluate(model, val_loader)
        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.2f}, Accuracy: {acc:.2f}%, Time: {duration:.1f}s")
        print(f"Exit 1: {exits[0]}, Exit 2: {exits[1]}, Exit 3: {exits[2]}, Final: {exits[3]}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
