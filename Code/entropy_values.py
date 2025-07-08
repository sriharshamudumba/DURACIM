import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING ===
def prepare_cifar100_subset(batch_size=16, max_samples=2000):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    subset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader

# === BACKBONE ===
def load_backbone():
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 100)
    ckpt = "checkpoints/resnet50_backbone.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model

# === ENTROPY ===
def entropy(logits):
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

# === MODEL ===
class BranchyResNet(nn.Module):
    def __init__(self, backbone, exit_blocks, num_classes=100):
        super().__init__()
        self.initial_layers = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList(list(backbone.layer1) + list(backbone.layer2) + list(backbone.layer3) + list(backbone.layer4))
        self.exit_blocks = exit_blocks
        self.feature_dims = [256]*3 + [512]*4 + [1024]*6 + [2048]*3
        self.exits = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.feature_dims[idx], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            ) for idx in exit_blocks
        ])
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        outputs = []
        x = self.initial_layers(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.exit_blocks:
                outputs.append(self.exits[self.exit_blocks.index(i)](x))
        final_out = self.final_classifier(x)
        return outputs, final_out

# === ENTROPY PLOTTER ===
def plot_entropy_distribution(model_configs, loader, save_path="entropy_combined_sets.png"):
    plt.figure(figsize=(14, 7))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for set_idx, (set_name, exit_blocks) in enumerate(model_configs.items()):
        backbone = load_backbone().to(device)
        model = BranchyResNet(backbone, exit_blocks).to(device)
        model.eval()

        entropies_per_exit = [[] for _ in exit_blocks]

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                outputs, _ = model(images)
                for i, out in enumerate(outputs):
                    ent = entropy(out).cpu().numpy()
                    entropies_per_exit[i].extend(ent)

        for i, entropies in enumerate(entropies_per_exit):
            label = f"{set_name} - Exit@{exit_blocks[i]}"
            plt.hist(entropies, bins=60, alpha=0.5, label=label, color=colors[i % len(colors)], density=True)

    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.title("Entropy Distribution per Exit Branch (All Sets, CIFAR-100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved entropy distribution plot to {save_path}")

# === MAIN ===
if __name__ == "__main__":
    test_loader = prepare_cifar100_subset(max_samples=2000)
    model_sets = {
        "Set1": [3],
        "Set2": [2, 10],
        "Set3": [3, 7, 11],
        "Set4": [2, 5, 9, 13]
    }
    plot_entropy_distribution(model_sets, test_loader)

