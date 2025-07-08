import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# === DATA LOADING ===
def prepare_cifar100(batch_size=16):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# === BACKBONE ===
def load_backbone():
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 100)
    checkpoint_path = "checkpoints/resnet50_backbone.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return model

# === METRICS ===
def entropy(x):
    probs = torch.softmax(x, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def submodel_parameters(model, exit_idx):
    count = count_parameters(model.initial_layers)
    for i in range(model.exit_blocks[exit_idx] + 1):
        count += count_parameters(model.blocks[i])
    count += count_parameters(model.exits[exit_idx])
    return count

# === MODEL ===
class BranchyResNet(nn.Module):
    def __init__(self, backbone, exit_blocks, num_classes=100):
        super().__init__()
        self.initial_layers = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList(
            list(backbone.layer1) + list(backbone.layer2) + list(backbone.layer3) + list(backbone.layer4)
        )
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

# === VALIDATION ===
def validate(model, loader, thresholds, forced_exit_idx=None):
    model.eval()
    num_exits = len(thresholds) + 1
    correct = [0] * (num_exits + 1)
    counts = [0] * (num_exits + 1)
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, final_out = model(images)
            batch_size = images.size(0)

            if forced_exit_idx is not None:
                if forced_exit_idx < len(outputs):
                    preds = outputs[forced_exit_idx].argmax(dim=1)
                else:
                    preds = final_out.argmax(dim=1)
                correct_idx = (preds == labels).sum().item()
                correct[forced_exit_idx] += correct_idx
                counts[forced_exit_idx] += batch_size
                total_correct += correct_idx
                total += batch_size
            else:
                exits = torch.full((batch_size,), num_exits, dtype=torch.int, device=device)
                for i, threshold in enumerate(thresholds):
                    if i < len(outputs):
                        ent = entropy(outputs[i])
                        exits[(exits == num_exits) & (ent < threshold)] = i

                preds = torch.empty_like(labels)
                for i in range(len(outputs)):
                    idx = exits == i
                    if idx.any():
                        preds[idx] = outputs[i][idx].argmax(dim=1)
                idx = exits == num_exits
                if idx.any():
                    preds[idx] = final_out[idx].argmax(dim=1)

                for i in range(num_exits + 1):
                    idx = exits == i
                    if idx.any():
                        correct[i] += (preds[idx] == labels[idx]).sum().item()
                        counts[i] += idx.sum().item()
                total_correct += (preds == labels).sum().item()
                total += batch_size

    exit_acc_only = [100 * correct[i] / counts[i] if counts[i] > 0 else 0 for i in range(num_exits + 1)]
    exit_acc_whole = [100 * correct[i] / total if counts[i] > 0 else 0 for i in range(num_exits + 1)]
    fractions = [counts[i] / total for i in range(num_exits + 1)]
    return {
        'accuracy': 100 * total_correct / total,
        'exit_counts': counts,
        'exit_accuracies_only': exit_acc_only,
        'exit_accuracies_whole': exit_acc_whole,
        'fractions': fractions
    }

# === WEIGHTED LOSS FUNCTION ===
def compute_weighted_loss(outputs, final_out, labels, criterion):
    N = len(outputs)
    weights = [0.5 / N] * N
    final_weight = 0.5
    losses = [weights[i] * criterion(out, labels) for i, out in enumerate(outputs)]
    losses.append(final_weight * criterion(final_out, labels))
    return sum(losses)

# === TRIAL ===
def run_trial(exit_blocks, thresholds, trial_name, regenerate_forced=False):
    result_path = f"results/{trial_name}.txt"
    retrain_required = not os.path.exists(result_path)

    train_loader, test_loader = prepare_cifar100()
    backbone = load_backbone().to(device)
    model = BranchyResNet(backbone, exit_blocks).to(device)

    if retrain_required:
        print(f"Training Trial: {trial_name}")
        for param in model.initial_layers.parameters(): param.requires_grad = False
        for param in model.blocks.parameters(): param.requires_grad = False
        for param in model.exits.parameters(): param.requires_grad = True
        for param in model.final_classifier.parameters(): param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(20):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    outputs, final_out = model(images)
                    loss = compute_weighted_loss(outputs, final_out, labels, criterion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            print(f"Epoch {epoch+1} | Loss: {running_loss:.2f}")

        results = validate(model, test_loader, thresholds)

        os.makedirs("results", exist_ok=True)
        with open(result_path, "w") as f:
            f.write(f"Trial: {trial_name}\n")
            f.write(f"Exit positions (after blocks): {exit_blocks}\n")
            f.write(f"Thresholds used: {thresholds}\n\n")

            for i in range(len(results['exit_counts'])):
                is_final = i == len(results['exit_counts']) - 1
                f.write(f"Exit {i+1} {'(Final exit if last)' if is_final else ''}:\n")
                f.write(f"- Samples exited: {results['exit_counts'][i]} ({results['fractions'][i]*100:.2f}%)\n")
                f.write(f"- Exit Accuracy (only exited samples): {results['exit_accuracies_only'][i]:.2f}%\n")
                f.write(f"- Exit Accuracy (wrt whole testset): {results['exit_accuracies_whole'][i]:.2f}%\n")
                if is_final:
                    final_params = count_parameters(model.final_classifier)
                    sub_params = count_parameters(model.initial_layers) + sum(count_parameters(b) for b in model.blocks) + final_params
                    f.write(f"- Exit Parameters: {final_params:,}\n")
                    f.write(f"- Submodel Parameters: {sub_params:,}\n\n")
                elif i < len(model.exits):
                    exit_params = count_parameters(model.exits[i])
                    sub_params = submodel_parameters(model, i)
                    f.write(f"- Exit Parameters: {exit_params:,}\n")
                    f.write(f"- Submodel Parameters: {sub_params:,}\n\n")

            total_params = count_parameters(model)
            f.write(f"Total Model Parameters (including exits + backbone): {total_params:,}\n")
            f.write(f"Overall Early-Exit Accuracy: {results['accuracy']:.2f}%\n")

    if regenerate_forced or retrain_required:
        with open(result_path, "r") as f:
            lines = f.readlines()

        with open(result_path, "w") as f:
            in_forced_section = False
            for line in lines:
                if line.strip().startswith("=== Forced Exit Accuracies"):
                    in_forced_section = True
                if not in_forced_section:
                    f.write(line)

            f.write(f"\n=== Forced Exit Accuracies (all samples forced through each exit) ===\n")
            for i in range(len(exit_blocks) + 1):
                forced_result = validate(model, test_loader, thresholds, forced_exit_idx=i)
                f.write(f"Forced Exit {i+1}: {forced_result['accuracy']:.2f}%\n")

# === RUNNER ===
if __name__ == "__main__":
    print("Running all trials for Early-Exit ResNet50 Improved...")

    for block_idx in range(16):
        run_trial([block_idx], [0.6], f"set1_single_exit_block_{block_idx}", regenerate_forced=True)

    for first in range(2):
        for second in range(4, 15):
            run_trial([first, second], [0.6, 1.0], f"set2_exit_blocks_{first}_{second}", regenerate_forced=True)

    run_trial([3, 7, 11], [0.6, 1.0, 1.4], "set3_three_evenly_spaced_exits", regenerate_forced=True)
    run_trial([2, 5, 9, 13], [0.6, 1.0, 1.4, 1.8], "set4_four_evenly_spaced_exits", regenerate_forced=True)

    print("\n All Trial Runs Completed!")

