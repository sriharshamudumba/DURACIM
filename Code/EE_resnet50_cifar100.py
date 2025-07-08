import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import os

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# === DATA LOADING ===
def prepare_cifar100(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2761))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# === BRANCHY RESNET50 ===
class BranchyResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        base.fc = nn.Identity()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.exit2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, num_classes))
        self.final_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        out1 = self.exit1(x)
        x = self.layer3(x)
        out2 = self.exit2(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        out_final = self.final_fc(x)
        return out1, out2, out_final

# === TRAIN FUNCTION ===
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    correct1 = correct2 = correct_final = 0
    loss1 = loss2 = loss_final = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            out1, out2, final = model(images)
            l1 = criterion(out1, labels)
            l2 = criterion(out2, labels)
            l3 = criterion(final, labels)
            total_loss = 0.3 * l1 + 0.3 * l2 + 0.4 * l3

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += labels.size(0)
        correct1 += (out1.argmax(1) == labels).sum().item()
        correct2 += (out2.argmax(1) == labels).sum().item()
        correct_final += (final.argmax(1) == labels).sum().item()
        loss1 += l1.item()
        loss2 += l2.item()
        loss_final += l3.item()

    return {
        "train_exit1_acc": 100 * correct1 / total,
        "train_exit2_acc": 100 * correct2 / total,
        "train_final_acc": 100 * correct_final / total,
        "train_exit1_loss": loss1 / len(loader),
        "train_exit2_loss": loss2 / len(loader),
        "train_final_loss": loss_final / len(loader),
    }

# === VALIDATION FUNCTION ===
def validate(model, loader, criterion):
    model.eval()
    correct1 = correct2 = correct_final = 0
    loss1 = loss2 = loss_final = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out1, out2, final = model(images)
            l1 = criterion(out1, labels)
            l2 = criterion(out2, labels)
            l3 = criterion(final, labels)

            total += labels.size(0)
            correct1 += (out1.argmax(1) == labels).sum().item()
            correct2 += (out2.argmax(1) == labels).sum().item()
            correct_final += (final.argmax(1) == labels).sum().item()
            loss1 += l1.item()
            loss2 += l2.item()
            loss_final += l3.item()

    return {
        "val_exit1_acc": 100 * correct1 / total,
        "val_exit2_acc": 100 * correct2 / total,
        "val_final_acc": 100 * correct_final / total,
        "val_exit1_loss": loss1 / len(loader),
        "val_exit2_loss": loss2 / len(loader),
        "val_final_loss": loss_final / len(loader),
    }

# === EVALUATE EXIT COUNTS ===
def evaluate_exit_distribution(model, loader, threshold1=2.5, threshold2=3.0):
    model.eval()
    exit1_count = exit2_count = final_count = 0
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            out1, out2, final = model(images)

            probs1 = torch.softmax(out1, dim=1)
            entropy1 = -torch.sum(probs1 * torch.log(probs1 + 1e-6), dim=1)

            probs2 = torch.softmax(out2, dim=1)
            entropy2 = -torch.sum(probs2 * torch.log(probs2 + 1e-6), dim=1)

            exits = torch.full((images.size(0),), 3, dtype=torch.int, device=device)
            confident1 = entropy1 < threshold1
            exits[confident1] = 1

            still_remaining = exits == 3
            confident2 = entropy2[still_remaining] < threshold2
            idx_remaining = torch.where(still_remaining)[0]
            exits[idx_remaining[confident2]] = 2

            exit1_count += (exits == 1).sum().item()
            exit2_count += (exits == 2).sum().item()
            final_count += (exits == 3).sum().item()
            total_samples += images.size(0)

    print("\n=== Exit Distribution ===")
    print(f"Exit 1: {exit1_count} ({exit1_count/total_samples*100:.2f}%)")
    print(f"Exit 2: {exit2_count} ({exit2_count/total_samples*100:.2f}%)")
    print(f"Final : {final_count} ({final_count/total_samples*100:.2f}%)")

# === FINAL INFERENCE TIME ===
def final_inference(model, loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            model(images)
    end_time = time.time()
    print(f"\nFinal Inference Time: {end_time - start_time:.2f} seconds")

# === MAIN ===
if __name__ == "__main__":
    train_loader, test_loader = prepare_cifar100()

    model = BranchyResNet50().to(device)
    ckpt_path = "branchy_resnet50_cifar100.pth"

    if os.path.exists(ckpt_path):
        print(f"\n✅ Model found! Loading from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("\n🚀 Training model from scratch...")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler()

        for epoch in range(1, 21):
            train_metrics = train(model, train_loader, optimizer, criterion, scaler)
            val_metrics = validate(model, test_loader, criterion)

            print(f"\nEpoch {epoch}/20")
            print(f"Train Exit1 Acc: {train_metrics['train_exit1_acc']:.2f}%, Exit2 Acc: {train_metrics['train_exit2_acc']:.2f}%, Final Acc: {train_metrics['train_final_acc']:.2f}%")
            print(f"Val   Exit1 Acc: {val_metrics['val_exit1_acc']:.2f}%, Exit2 Acc: {val_metrics['val_exit2_acc']:.2f}%, Final Acc: {val_metrics['val_final_acc']:.2f}%")
            print("-" * 70)

        print(f"\n💾 Saving model to: {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

    # Inference phase
    final_inference(model, test_loader)
    evaluate_exit_distribution(model, test_loader)

