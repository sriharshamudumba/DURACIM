import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load the combined_results.txt
with open("S:\Edu\MS Thesis\Early exit heteroPIM\EarlyExit\Code\combined_results.txt", "r") as file:
    lines = file.readlines()

# Extract parameters for Set2 with and without threshold
set2_data = []
trial_name = ""
exit_positions = []
thresholded_params = []
forced_params = []

capture_forced = False
current_forced = []

for line in lines:
    line = line.strip()
    if line.startswith("Trial: set2_"):
        if trial_name and thresholded_params:
            set2_data.append((trial_name, exit_positions.copy(), thresholded_params.copy(), forced_params.copy()))
            thresholded_params.clear()
            forced_params.clear()
            exit_positions.clear()
        trial_name = line.split(": ")[1]
        capture_forced = False
        current_forced = []
    elif line.startswith("Exit positions"):
        exit_positions = eval(line.split(": ")[1])
    elif "Submodel Parameters" in line:
        val = int(line.split(": ")[1].replace(",", ""))
        thresholded_params.append(val)
    elif line.startswith("=== Forced Exit Accuracies"):
        capture_forced = True
        current_forced = []
    elif capture_forced and line.startswith("Forced Exit"):
        forced_params.append(0)  # Placeholder to align lengths; we do not have actual forced submodule param

# Final append
if trial_name and thresholded_params:
    set2_data.append((trial_name, exit_positions.copy(), thresholded_params.copy(), forced_params.copy()))

# Convert to dataframe
records = []
for trial, blocks, thresh_vals, forced_vals in set2_data:
    for i in range(len(blocks)):
        records.append({
            "Trial": trial,
            "Exit": f"Exit{i+1}",
            "Block": blocks[i],
            "Thresholded_Params": thresh_vals[i] if i < len(thresh_vals) else 0,
            "Forced_Params": forced_vals[i] if i < len(forced_vals) else 0
        })

df = pd.DataFrame(records)

# Create grouped bar plot with 3 bars per exit: Thresholded vs Forced per exit
exit1 = df[df["Exit"] == "Exit1"]
exit2 = df[df["Exit"] == "Exit2"]

fig, ax = plt.subplots(figsize=(16, 6))

bar_width = 0.25
x = range(len(exit1))

ax.bar([p - bar_width/2 for p in x], exit1["Thresholded_Params"], bar_width, label="Exit1 Thresholded")
ax.bar([p + bar_width/2 for p in x], exit2["Thresholded_Params"], bar_width, label="Exit2 Thresholded")

ax.set_xticks(x)
ax.set_xticklabels(exit1["Trial"], rotation=90)
ax.set_ylabel("Submodule Parameters")
ax.set_title("Set2: Submodule Parameters for Each Exit (Thresholded Only)")
ax.legend()
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
