import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

from torch.optim.lr_scheduler import CosineAnnealingLR

# === DEVICE SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === DATA LOADING ===
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2761))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2761))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),
    batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

# === MODEL SETUP ===
# Load pretrained ResNet50
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# === LOSS, OPTIMIZER, SCHEDULER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)  # smooth LR decay

scaler = torch.cuda.amp.GradScaler()  # for AMP

# === TRAINING FUNCTION ===
def train(model, optimizer, criterion, scheduler, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # AMP for speed
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/50] - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        scheduler.step()

# === EVALUATION FUNCTION ===
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    end_time = time.time()
    acc = 100 * correct / total
    inference_time = end_time - start_time
    return acc, inference_time

# === MAIN RUN ===
if __name__ == "__main__":
    train(model, optimizer, criterion, scheduler, epochs=50)
    acc, inf_time = evaluate(model)
    print(f"\n[ResNet50 Fine-tuned CIFAR-100] Accuracy: {acc:.2f}%, Inference Time: {inf_time:.2f}s")

