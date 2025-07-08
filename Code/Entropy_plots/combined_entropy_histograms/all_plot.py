import matplotlib.pyplot as plt
import pandas as pd
import re

# === Load and parse combined_results.txt ===
with open("combined_results.txt", "r") as f:
    content = f.read()

# Extract trial blocks
trial_blocks = re.findall(r"Trial:\s+(.*?)\n(.*?)(?=\n=|$)", content, re.DOTALL)

# Extract info from each exit
records = []
for trial_name, block in trial_blocks:
    matches = re.findall(
        r"Exit\s+(\d+):\s*"
        r"- Samples exited: (\d+).*?"
        r"- Exit Accuracy \(only exited samples\): ([\d.]+)%.*?"
        r"- Exit Parameters: ([\d,]+)", 
        block, re.DOTALL
    )
    for exit_idx, samples, accuracy, params in matches:
        records.append({
            "Trial": trial_name.strip(),
            "Exit": int(exit_idx),
            "SamplesExited": int(samples),
            "Accuracy": float(accuracy),
            "Parameters": int(params.replace(",", ""))
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# === Plot ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["SamplesExited"], df["Accuracy"],
    c=df["Parameters"], cmap="plasma", s=100, edgecolors='k', alpha=0.8
)

plt.xlabel("Samples Exited")
plt.ylabel("Accuracy (%)")
plt.title("Samples Exited vs Accuracy (colored by Model Parameters)")
cbar = plt.colorbar(scatter)
cbar.set_label("Model Parameters")
plt.grid(True)
plt.tight_layout()
plt.savefig("samples_vs_accuracy_colored_by_parameters.png")
plt.show()

print("✅ Saved: samples_vs_accuracy_colored_by_parameters.png")
