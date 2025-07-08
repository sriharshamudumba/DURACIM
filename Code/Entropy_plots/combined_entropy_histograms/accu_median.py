import torch
import matplotlib.pyplot as plt
import pandas as pd
import re
from statistics import median

# === Load entropy data ===
entropy_set1 = torch.load("entropy_set1.pt", map_location="cpu")
entropy_set2 = torch.load("entropy_set2.pt", map_location="cpu")

# === Combine both sets ===
combined_entropy = {**entropy_set1, **entropy_set2}

# === Compute median entropy for each exit ===
exit_medians = {}

for name, values in combined_entropy.items():
    if hasattr(values, "tolist"):
        values = values.tolist()
    # Extract the exit index from something like "Set1_Exit0"
    match = re.search(r'Exit(\d+)', name)
    if match:
        exit_index = int(match.group(1))
        exit_medians[exit_index] = median(values)

# === Load and parse combined_results.txt ===
with open("combined_results.txt", "r") as f:
    content = f.read()

trial_blocks = re.findall(r"Trial:\s+(.*?)\n(.*?)(?=\n=|$)", content, re.DOTALL)

records = []
for trial_name, block in trial_blocks:
    exits = re.findall(
        r"Exit\s+\d+.*?- Exit Accuracy \(only exited samples\): ([\d.]+)%.*?"
        r"- Exit Parameters: ([\d,]+)", block, re.DOTALL)
    for idx, (acc, params) in enumerate(exits):
        records.append({
            "Trial": trial_name.strip(),
            "Exit": idx,
            "Accuracy": float(acc),
            "Parameters": int(params.replace(',', ''))
        })

df = pd.DataFrame(records)

# === Map median entropy to each exit ===
df["MedianEntropy"] = df["Exit"].map(exit_medians)

# === Plot ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["MedianEntropy"], df["Accuracy"],
    c=df["Parameters"], cmap="viridis", s=80, edgecolors='k'
)

plt.xlabel("Median Entropy")
plt.ylabel("Exit Accuracy (%)")
plt.title("Submodel Accuracy vs Median Entropy (colored by Parameters)")
plt.colorbar(scatter, label="Number of Parameters")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_entropy_vs_parameters.png")
plt.show()

print(" Plot saved as 'accuracy_vs_entropy_vs_parameters.png'")
