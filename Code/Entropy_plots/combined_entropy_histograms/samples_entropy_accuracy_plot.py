import torch
import matplotlib.pyplot as plt
import pandas as pd
import re
from statistics import median

# === Step 1: Load entropy data ===
entropy_set1 = torch.load("entropy_set1.pt", map_location="cpu")
entropy_set2 = torch.load("entropy_set2.pt", map_location="cpu")
combined_entropy = {**entropy_set1, **entropy_set2}

# === Step 2: Compute median entropy per exit ===
exit_medians = {}
for name, values in combined_entropy.items():
    if hasattr(values, "tolist"):
        values = values.tolist()
    match = re.search(r'Exit(\d+)', name)
    if match:
        exit_index = int(match.group(1))
        exit_medians[exit_index] = median(values)

# === Step 3: Load and parse combined_results.txt ===
with open("combined_results.txt", "r") as f:
    content = f.read()

# Extract blocks for each trial
trial_blocks = re.findall(r"Trial:\s+(.*?)\n(.*?)(?=\n=|$)", content, re.DOTALL)

# Extract samples, accuracy, and include exit index
records = []
for trial_name, block in trial_blocks:
    exits = re.findall(
        r"Exit\s+(\d+):\s*- Samples exited: (\d+) .*?"
        r"- Exit Accuracy \(only exited samples\): ([\d.]+)%", block, re.DOTALL
    )
    for exit_idx_str, samples, acc in exits:
        exit_idx = int(exit_idx_str)
        records.append({
            "Trial": trial_name.strip(),
            "Exit": exit_idx,
            "SamplesExited": int(samples),
            "Accuracy": float(acc)
        })

# === Step 4: Create DataFrame and map median entropy ===
df = pd.DataFrame(records)
df["MedianEntropy"] = df["Exit"].map(exit_medians)

# === Step 5: Plot ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["MedianEntropy"], df["Accuracy"],
    c=df["SamplesExited"], cmap="plasma", s=80, edgecolors='k'
)

plt.xlabel("Median Entropy")
plt.ylabel("Exit Accuracy (%)")
plt.title("Samples Exited vs Median Entropy vs Accuracy")
plt.colorbar(scatter, label="Samples Exited")
plt.grid(True)
plt.tight_layout()
plt.savefig("samples_vs_entropy_vs_accuracy.png")
plt.show()

print("Plot saved as 'samples_vs_entropy_vs_accuracy.png'")
