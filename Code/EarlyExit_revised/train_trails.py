# Required Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import gc

# === Config ===
BACKBONE_PATH = "/research/duwe/Sri/Code/checkpoints/resnet50_backbone.pth"
RESULTS_DIR = "/research/duwe/Sri/EarlyExit_revised/results"
ENTROPY_DIR = "./entropy_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENTROPY_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CIFAR-100 ===
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
train_set = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# === Utils ===
def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# === Exit Wrapper ===
class BranchyResNet(nn.Module):
    def __init__(self, backbone, exit_blocks):
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layers = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.exit_blocks = exit_blocks
        self.exit_branches = nn.ModuleList()

        block_to_layer = [0]*3 + [1]*4 + [2]*6 + [3]*3
        self.block_to_layer = block_to_layer

        self.exit_points = []
        for blk in exit_blocks:
            layer_idx = block_to_layer[blk]
            channels = [256, 512, 1024, 2048][layer_idx]
            self.exit_points.append((layer_idx, blk))
            self.exit_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 100)
            ))

    def forward_until(self, x, block_idx):
        x = self.stem(x)
        blk_count = 0
        for i, layer in enumerate(self.layers):
            for j, blk in enumerate(layer):
                x = blk(x)
                if blk_count == block_idx:
                    return x
                blk_count += 1
        return x

# === Training and Evaluation ===
def train_exit_branch(model, exit_idx, exit_block, epochs=10):
    branch = model.exit_branches[exit_idx]
    for param in model.parameters(): param.requires_grad = False
    for param in branch.parameters(): param.requires_grad = True
    optimizer = torch.optim.Adam(branch.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    branch.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad(): feats = model.forward_until(images, exit_block)
            logits = branch(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# === Evaluate and Log ===
def evaluate(model, trail_name):
    model.eval()
    all_outputs = [[] for _ in model.exit_branches]
    all_entropies = [[] for _ in model.exit_branches]
    all_preds = [[] for _ in model.exit_branches]
    labels_all = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_all.extend(labels.cpu().numpy())
            for i, (blk, branch) in enumerate(zip(model.exit_blocks, model.exit_branches)):
                feats = model.forward_until(images, blk)
                logits = branch(feats)
                all_outputs[i].append(logits.cpu())
                all_entropies[i].append(compute_entropy(logits).cpu())
                all_preds[i].append(torch.argmax(logits, dim=1).cpu())

    results = {}
    labels_all = np.array(labels_all)
    total_correct = 0
    for i in range(len(model.exit_branches)):
        logits_all = torch.cat(all_outputs[i])
        ent_all = torch.cat(all_entropies[i]).numpy()
        preds_all = torch.cat(all_preds[i]).numpy()
        acc = 100 * np.mean(preds_all == labels_all)
        total_correct += np.sum(preds_all == labels_all)

        entropy_path = os.path.join(ENTROPY_DIR, f"{trail_name}_exit{i+1}_entropy.txt")
        preds_path = os.path.join(ENTROPY_DIR, f"{trail_name}_exit{i+1}_preds.txt")
        np.savetxt(entropy_path, ent_all, fmt="%.5f")
        np.savetxt(preds_path, preds_all, fmt="%d")

        results[f"Exit_{i+1}"] = {
            "accuracy": acc,
            "forced_accuracy": acc,
            "num_exited": len(preds_all),
            "parameters": count_parameters(model.exit_branches[i])
        }

    overall_acc = 100 * total_correct / (len(labels_all) * len(model.exit_branches))
    results["Overall"] = {
        "early_exit_accuracy": overall_acc
    }

    with open(os.path.join(RESULTS_DIR, f"{trail_name}.txt"), 'w') as f:
        json.dump(results, f, indent=2)

# === Trail Sets ===
def run_all_trails():
    def skip_if_exists(trail_name):
        return os.path.exists(os.path.join(RESULTS_DIR, f"{trail_name}.txt"))

    # Set 1
    for blk in range(16):
        trail_name = f"set1_block{blk}"
        if skip_if_exists(trail_name):
            print(f"Skipping {trail_name} (already done)")
            continue
        resnet50 = torchvision.models.resnet50()
        resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)
        resnet50.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu"))
        model = BranchyResNet(resnet50, [blk]).to(device)
        train_exit_branch(model, 0, blk)
        evaluate(model, trail_name)
        del model, resnet50
        torch.cuda.empty_cache()
        gc.collect()

    # Set 2
    for b1 in range(0, 5):
        for b2 in range(b1 + 5, 15):
            trail_name = f"set2_b1_{b1}_b2_{b2}"
            if skip_if_exists(trail_name):
                print(f"Skipping {trail_name} (already done)")
                continue
            resnet50 = torchvision.models.resnet50()
            resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)
            resnet50.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu"))
            model = BranchyResNet(resnet50, [b1, b2]).to(device)
            train_exit_branch(model, 0, b1)
            train_exit_branch(model, 1, b2)
            evaluate(model, trail_name)
            del model, resnet50
            torch.cuda.empty_cache()
            gc.collect()

    # Set 3
    trail_name = "set3"
    if not skip_if_exists(trail_name):
        resnet50 = torchvision.models.resnet50()
        resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)
        resnet50.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu"))
        model = BranchyResNet(resnet50, [5, 10, 15]).to(device)
        for i, blk in enumerate([5, 10, 15]):
            train_exit_branch(model, i, blk)
        evaluate(model, trail_name)
        del model, resnet50
        torch.cuda.empty_cache()
        gc.collect()

    # Set 4
    trail_name = "set4"
    if not skip_if_exists(trail_name):
        resnet50 = torchvision.models.resnet50()
        resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)
        resnet50.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu"))
        model = BranchyResNet(resnet50, [4, 8, 12, 15]).to(device)
        for i, blk in enumerate([4, 8, 12, 15]):
            train_exit_branch(model, i, blk)
        evaluate(model, trail_name)
        del model, resnet50
        torch.cuda.empty_cache()
        gc.collect()

# Run all trails
run_all_trails()
print("All trails completed.")

