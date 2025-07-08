import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import json

# ========== Load baseline ResNet50 ==========
baseline_model = tf.keras.models.load_model("Hetero-PIM/baseline_resnet50_tf_75p8_top1acc.h5")

NUM_LAYERS = 50

# get layer sizes (only Conv2D and Dense)
layer_sizes = []
for layer in baseline_model.layers:
    try:
        params = np.sum([np.prod(w.shape) for w in layer.weights])
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer_sizes.append(params)
    except:
        continue

# pad to NUM_LAYERS if needed
while len(layer_sizes) < NUM_LAYERS:
    layer_sizes.append(1)

# budgets
BUDGET_LATENCY = 1e6
BUDGET_AREA = 1e4
BUDGET_ENERGY = 1e5
BASELINE_ACCURACY = 75.8
CROSSBAR_SIZE = 128

# IRDS parameters
MEMORY_TYPES = {
    "SRAM": {
        "read_ns": 50,
        "endurance": 1e15,
        "energy_pj": 1,
        "area": 1
    },
    "ReRAM": {
        "read_ns": 15,
        "endurance": 1e9,
        "energy_pj": 0.5,
        "area": 0.5
    }
}

# write frequency:
# ReRAM layers 0–41 frozen (0 writes)
# SRAM layers 42–49 with lower write frequency 0.1
write_freqs = np.zeros(NUM_LAYERS)
for i in range(NUM_LAYERS):
    if i >= 42:
        write_freqs[i] = 0.1
    else:
        write_freqs[i] = 0.0

# ========== DURACIM environment ==========
class DURACIMEnv:
    def __init__(self):
        self.reset()
    def reset(self):
        self.exit_layer = 0
        return self._get_state()
    def _get_state(self):
        return np.array([self.exit_layer / NUM_LAYERS])
    def step(self, action):
        self.exit_layer = action
        done = True
        reward = self._compute_reward()
        return self._get_state(), reward, done
    def _compute_reward(self):
        latency = 0
        area = 0
        energy = 0
        lifetimes = []
        for i in range(NUM_LAYERS):
            mem = "ReRAM" if i < 42 else "SRAM"
            memprops = MEMORY_TYPES[mem]
            n_crossbars = int(np.ceil(layer_sizes[i] / CROSSBAR_SIZE))
            latency += memprops["read_ns"] * n_crossbars
            energy += memprops["energy_pj"] * n_crossbars
            area += memprops["area"] * n_crossbars
            wf = write_freqs[i]
            if wf > 0:
                lifetime = memprops["endurance"] / wf
            else:
                lifetime = 1e20  # effectively infinite
            lifetimes.append(lifetime)
        system_lifetime = min(lifetimes)
        norm_latency = latency / BUDGET_LATENCY
        norm_area = area / BUDGET_AREA
        norm_energy = energy / BUDGET_ENERGY
        exit_bonus = (NUM_LAYERS - self.exit_layer) / NUM_LAYERS
        reward = (
            np.log1p(system_lifetime)*(0.5 + 0.5*exit_bonus)
            - 0.5 * norm_latency
            - 0.2 * norm_area
            - 0.2 * norm_energy
        )
        self.last_latency = latency
        self.last_area = area
        self.last_energy = energy
        self.last_lifetime = system_lifetime
        return reward

# ========== REINFORCE agent ==========
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64, NUM_LAYERS)
        )
    def forward(self,x):
        return self.fc(x)

class REINFORCEAgent:
    def __init__(self, lr=1e-3):
        self.policy = PolicyNet()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
    def select_action(self, state):
        logits = self.policy(torch.tensor(state,dtype=torch.float32).unsqueeze(0)).squeeze()
        probs = torch.softmax(logits,dim=0)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / len(probs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()
    def update_policy(self, gamma=0.99):
        G=0
        returns=[]
        for r in reversed(self.rewards):
            G = r + gamma*G
            returns.insert(0,G)
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = -torch.sum(torch.stack(self.log_probs)*returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs=[]
        self.rewards=[]

# ========== Baseline metrics ==========
baseline_latency = 0
baseline_area = 0
baseline_energy = 0
baseline_lifetimes = []
for i in range(NUM_LAYERS):
    memprops = MEMORY_TYPES["SRAM"]
    n_crossbars = int(np.ceil(layer_sizes[i]/CROSSBAR_SIZE))
    baseline_latency += memprops["read_ns"] * n_crossbars
    baseline_energy += memprops["energy_pj"] * n_crossbars
    baseline_area += memprops["area"] * n_crossbars
    lifetime = memprops["endurance"] / 0.5  # same as SRAM write freq
    baseline_lifetimes.append(lifetime)
baseline_lifetime = min(baseline_lifetimes)

print(f"\n[Baseline ResNet50 (SRAM all layers)]")
print(f"Latency: {baseline_latency:.2f} ns")
print(f"Lifetime: {baseline_lifetime:.2e} cycles")
print(f"Area: {baseline_area:.2f}")
print(f"Energy: {baseline_energy:.2f} pJ")
print(f"Accuracy: {BASELINE_ACCURACY}%")

# ========== Training ==========
env = DURACIMEnv()
agent = REINFORCEAgent()
reward_history=[]
NUM_EPISODES=500

for ep in range(NUM_EPISODES):
    state = env.reset()
    total_reward=0
    while True:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        agent.rewards.append(reward)
        total_reward += reward
        if done:
            break
    agent.update_policy()
    reward_history.append(total_reward)
    print(f"Ep {ep+1}/{NUM_EPISODES} Reward: {total_reward:.3e}")

# ========== Best policy ==========
best_idx = int(np.argmax(reward_history))
best_exit = best_idx % NUM_LAYERS
env_best = DURACIMEnv()
env_best.exit_layer = best_exit
best_reward = env_best._compute_reward()

# ========== Report ==========
print("\n=============================")
print("[Best DURACIM Design Results]")
print("=============================")
print(f"Best Exit at Layer: {best_exit}")
print(f"Reward: {best_reward:.4e}")
print(f"Latency: {env_best.last_latency:.2f} ns")
print(f"Lifetime: {env_best.last_lifetime:.2e} cycles")
print(f"Area: {env_best.last_area:.2f}")
print(f"Energy: {env_best.last_energy:.2f} pJ")
print("Note: Accuracy to be measured after branch fine-tuning")

# ========== Comparison ==========
print("\n========== Comparison ==========")
print(f"{'Metric':<15} | {'Baseline':<20} | {'DURACIM':<20}")
print("-"*60)
print(f"{'Latency(ns)':<15} | {baseline_latency:<20.2f} | {env_best.last_latency:<20.2f}")
print(f"{'Lifetime':<15} | {baseline_lifetime:<20.2e} | {env_best.last_lifetime:<20.2e}")
print(f"{'Area':<15} | {baseline_area:<20.2f} | {env_best.last_area:<20.2f}")
print(f"{'Energy(pJ)':<15} | {baseline_energy:<20.2f} | {env_best.last_energy:<20.2f}")
print(f"{'Accuracy':<15} | {BASELINE_ACCURACY:<20.2f} | to be measured")
print("="*60)

# ========== Assumptions ==========
print("\n========== Assumptions and Calculations ==========")
print(f"- Baseline: ResNet50 pretrained on ImageNet, top-1 accuracy = {BASELINE_ACCURACY}%")
print("- SRAM: read 50ns, endurance 1e15 cycles, energy 1 pJ, area 1 unit")
print("- ReRAM: read 15ns, endurance 1e9 cycles, energy 0.5 pJ, area 0.5 unit")
print("- Crossbar size: 128 elements")
print("- Early layers (1–42) mapped to ReRAM with weights frozen (no writes after deployment)")
print("- Deeper layers (43–50) mapped to SRAM with 0.1 write frequency to reflect early exits")
print("- Latency, area, energy computed based on total crossbars")
print("- Lifetime computed per layer, then min() across all layers")
print("="*60)

# ========== Save ==========
best_design = {
    "exit_layer": int(best_exit),
    "reward": float(best_reward),
    "latency": float(env_best.last_latency),
    "lifetime": float(env_best.last_lifetime),
    "area": float(env_best.last_area),
    "energy": float(env_best.last_energy),
    "baseline_latency": float(baseline_latency),
    "baseline_lifetime": float(baseline_lifetime),
    "baseline_area": float(baseline_area),
    "baseline_energy": float(baseline_energy)
}
with open("resnet50_rl_duracim_irds.json","w") as f:
    json.dump(best_design, f, indent=2)
