import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entropy(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def prepare_cifar100(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# === MODEL ===
class BranchyResNet(nn.Module):
    def __init__(self, backbone, exit_blocks):
        super().__init__()
        self.initial_layers = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList(list(backbone.layer1) + list(backbone.layer2) +
                                    list(backbone.layer3) + list(backbone.layer4))
        self.feature_dims = [256]*3 + [512]*4 + [1024]*6 + [2048]*3
        self.exit_blocks = exit_blocks
        self.exits = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.feature_dims[b], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 100)
            ) for b in exit_blocks
        ])
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 100)
        )

    def forward(self, x):
        x = self.initial_layers(x)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.exit_blocks:
                idx = self.exit_blocks.index(i)
                out = self.exits[idx](x)
                outputs.append(out)
        outputs.append(self.final_classifier(x))
        return outputs

def load_backbone():
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 100)
    ckpt_path = "checkpoints/resnet50_backbone.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model

# === COLLECT ENTROPY DISTRIBUTIONS ===
def collect_entropy_distributions(model, dataloader, tag):
    model.eval()
    entropies = defaultdict(list)
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=tag):
            images = images.to(device)
            outputs = model(images)
            for i, out in enumerate(outputs):
                ent = entropy(out).cpu().tolist()
                entropies[f"{tag}_Exit{i}"].extend(ent)
    return entropies

# === MAIN ===
if __name__ == "__main__":
    os.chdir("/research/duwe/Sri/Code")
    test_loader = prepare_cifar100()

    # === Set1 ===
    set1_data = {}
    for i in range(16):
        backbone = load_backbone()
        model = BranchyResNet(backbone, exit_blocks=[i]).to(device)
        entropies = collect_entropy_distributions(model, test_loader, f"Set1")
        set1_data[f"Set1_Exit{i}"] = entropies[f"Set1_Exit0"]  # always first exit
    torch.save(set1_data, "entropy_set1.pt")
    print("✅ Set1 entropy values saved to entropy_set1.pt")

    # === Set2 ===
    set2_data = {}
    for i in range(4, 15):
        for prefix in [0, 1, 2, 3]:
            tag = f"Set2_{prefix}_{i}"
            backbone = load_backbone()
            model = BranchyResNet(backbone, exit_blocks=[prefix, i]).to(device)
            entropies = collect_entropy_distributions(model, test_loader, tag)
            set2_data[f"Set2_Exit1_{prefix}_{i}"] = entropies[f"{tag}_Exit0"]
            set2_data[f"Set2_Exit2_{prefix}_{i}"] = entropies[f"{tag}_Exit1"]
    torch.save(set2_data, "entropy_set2.pt")
    print("✅ Set2 entropy values saved to entropy_set2.pt")

