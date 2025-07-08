import re
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load file
with open("combined_results.txt", "r") as f:
    content = f.read()

# Step 2: Regex to extract trial blocks
trial_pattern = re.compile(r"Trial:\s+(.*?)\n(.*?)(?=\n=|$)", re.DOTALL)
exit_pattern = re.compile(
    r"Exit\s+\d+\s*:\s*.*?- Exit Accuracy \(only exited samples\): ([\d.]+)%.*?"
    r"- Exit Parameters: ([\d,]+)", re.DOTALL
)

# Step 3: Extract structured data
trial_data = []
for trial_name, trial_text in trial_pattern.findall(content):
    exits = exit_pattern.findall(trial_text)
    for i, (acc, params) in enumerate(exits, start=1):
        trial_data.append({
            "Trial": trial_name.strip(),
            "Exit": f"Exit {i}",
            "Accuracy": float(acc),
            "Parameters": int(params.replace(',', ''))
        })

df = pd.DataFrame(trial_data)

# Step 4: Plot
plt.figure(figsize=(12, 7))
for trial_name, group in df.groupby("Trial"):
    plt.plot(group["Parameters"], group["Accuracy"], marker='o', label=trial_name)

plt.title("Accuracy vs Parameters (per Exit)")
plt.xlabel("Submodel Parameters")
plt.ylabel("Exit Accuracy (%)")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
plt.tight_layout()
plt.savefig("accuracy_vs_parameters.png")

print(" Plot saved as 'accuracy_vs_parameters.png'")
