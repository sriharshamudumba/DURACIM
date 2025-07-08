import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === CIFAR-100 TEST SET LOADER ===
def prepare_cifar100_test(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2761))
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

# === YOUR BRANCHYRESNET50 MODEL ===
class BranchyResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        base.fc = nn.Identity()

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1
        )
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
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

# === EVALUATE EXIT DISTRIBUTION ===
def evaluate_exit_distribution(model, test_loader, threshold1=2.5, threshold2=3.0):
    model.eval()
    exit1_count = 0
    exit2_count = 0
    final_count = 0
    total_samples = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            out1, out2, final = model(images)

            # Entropy calculations
            probs1 = torch.softmax(out1, dim=1)
            entropy1 = -torch.sum(probs1 * torch.log(probs1 + 1e-6), dim=1)

            probs2 = torch.softmax(out2, dim=1)
            entropy2 = -torch.sum(probs2 * torch.log(probs2 + 1e-6), dim=1)

            exits = torch.full((images.size(0),), 3, dtype=torch.int, device=device)  # Default: final exit

            confident1 = entropy1 < threshold1
            exits[confident1] = 1

            still_remaining = exits == 3
            confident2 = entropy2[still_remaining] < threshold2
            idx_remaining = torch.where(still_remaining)[0]
            exits[idx_remaining[confident2]] = 2

            # Count exits
            exit1_count += (exits == 1).sum().item()
            exit2_count += (exits == 2).sum().item()
            final_count += (exits == 3).sum().item()
            total_samples += images.size(0)

    print("\n=== Exit Branch Distribution ===")
    print(f"Total samples: {total_samples}")
    print(f"Exited at Exit 1: {exit1_count} samples ({100*exit1_count/total_samples:.2f}%)")
    print(f"Exited at Exit 2: {exit2_count} samples ({100*exit2_count/total_samples:.2f}%)")
    print(f"Exited at Final Exit: {final_count} samples ({100*final_count/total_samples:.2f}%)")

# === MAIN ===
if __name__ == "__main__":
    test_loader = prepare_cifar100_test(batch_size=128)

    # Load your already trained BranchyResNet50 model
    model = BranchyResNet50(num_classes=100).to(device)

    # Load your checkpoint if available (optional)
    # model.load_state_dict(torch.load("your_model_checkpoint.pth"))

    evaluate_exit_distribution(model, test_loader)

