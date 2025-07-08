import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load combined results.txt file
with open("EarlyExit/Code/combined_results.txt", "r") as f:
    lines = f.readlines()

# Extract entropy-related data
entropy_data = {}
current_trial = None
entropy_pattern = re.compile(r"Forced Exit \d+: ([\d.]+)%")

for line in lines:
    line = line.strip()
    if line.startswith("Trial:"):
        current_trial = line.split("Trial:")[1].strip()
        entropy_data[current_trial] = []
    elif entropy_pattern.search(line):
        accuracy = float(entropy_pattern.search(line).group(1))
        entropy_data[current_trial].append(accuracy)

# Build a DataFrame to store median entropy-like values for each set and trial
entropy_records = []

for trial, values in entropy_data.items():
    if not values:
        continue
    median_entropy = sorted(values)[len(values)//2]
    set_name = "Set1" if "set1_" in trial else "Set2" if "set2_" in trial else \
               "Set3" if "set3_" in trial else "Set4" if "set4_" in trial else "Other"
    entropy_records.append((set_name, trial, median_entropy))

df_entropy = pd.DataFrame(entropy_records, columns=["Set", "Trial", "Median_Entropy_Accuracy"])

# Plot histograms for each set
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so

unique_sets = df_entropy["Set"].unique()

fig, axes = plt.subplots(nrows=len(unique_sets), figsize=(10, 5 * len(unique_sets)))

for idx, set_name in enumerate(unique_sets):
    subset = df_entropy[df_entropy["Set"] == set_name]
    axes[idx].hist(subset["Median_Entropy_Accuracy"], bins=10, edgecolor="black")
    axes[idx].set_title(f"{set_name}: Median Forced Exit Accuracy Distribution")
    axes[idx].set_xlabel("Median Accuracy (%)")
    axes[idx].set_ylabel("Number of Trials")

plt.tight_layout()
plt.show()
