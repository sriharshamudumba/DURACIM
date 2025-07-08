import math
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import json

# ============= Added necessary spec values-Sohan ==============
xbar_size_x, xbar_size_y = 128, 128
datawidth = 16
bits_per_cell = 1
xbars_per_macro = 8
macros_per_tile = 8
tiles = 12*12
total_macro_isaac = macros_per_tile*tiles

cycle_time = 100e-9
cycles_per_mvm = 20
write_latencies = [0.5e-9,50e-9,10e-9,50e-9] # Dr Wang paper
# write_latencies = [1e-9,1000e-9,50e-9,10000000e-9] # IRDS
devices = ["SRAM","ReRAM","STTRAM","PCM"]
# =============================================================
# ============= Added Helper Functions-Sohan ==================
def next_multiple(x, size):
    return int(np.ceil(x / size))

def get_occupied_tiles(layer_rows,
                       layer_cols,
                       xbar_size_x = 128,
                       xbar_size_y = 128,
                       datawidth = 16,
                       bits_per_cell = 1,
                       xbars_per_macro = 8,
                       macros_per_tile = 8,
                       tiles = 12*12):
    
    total_macro_isaac = macros_per_tile*tiles
    
    macros_y = next_multiple(layer_rows, xbar_size_x)
    macros_x = next_multiple(layer_cols, xbar_size_y)
    
    occupied_macros = macros_y*macros_x*datawidth / (bits_per_cell*xbars_per_macro)
    occupied_tiles = np.ceil(occupied_macros/macros_per_tile)

    return occupied_tiles

def get_area(tiles, mem_tech):
    
    tile_area_mem = {
        "SRAM": {'router': 0.01875/4,
                'eDRAM_buf': 0.083,
                'inter_macro_network': 0.06,
                'shift_add': 0.011045280000000001,
                'output_quantize': 0.0415,
                'input_buffer': 0.014263360000000001,
                'output_buffer': 0.02967424,
                'adc': 0.41403328000000006,
                'row_drivers': 0.045100864000000004,
                'cim_unit': 0.28991029248},
        "ReRAM": {'router': 0.01875/4,
                'eDRAM_buf': 0.083,
                'inter_macro_network': 0.06,
                'shift_add': 0.011045280000000001,
                'output_quantize': 0.0415,
                'input_buffer': 0.014263360000000001,
                'output_buffer': 0.02967424,
                'adc': 0.41403328000000006,
                'row_drivers': 0.09143103999999999,
                'cim_unit': 0.028991029247999997},
        "STTRAM": {'router': 0.01875/4,
                'eDRAM_buf': 0.083,
                'inter_macro_network': 0.06,
                'shift_add': 0.011045280000000001,
                'output_quantize': 0.0415,
                'input_buffer': 0.014263360000000001,
                'output_buffer': 0.02967424,
                'adc': 0.41403328000000006,
                'row_drivers': 0.08797952,
                'cim_unit': 0.10737418240000002},
        "PCM": {'router': 0.01875/4,
                'eDRAM_buf': 0.083,
                'inter_macro_network': 0.06,
                'shift_add': 0.011045280000000001,
                'output_quantize': 0.0415,
                'input_buffer': 0.014263360000000001,
                'output_buffer': 0.02967424,
                'adc': 0.41403328000000006,
                'row_drivers': 0.09143103999999999,
                'cim_unit': 0.028991029247999997}
        }
    
    area = tiles*(sum(tile_area_mem[mem_tech].values()))
    return area

def compute_conv2d_output(h, w, kernel, stride, padding):
    h_out = math.floor((h + 2 * padding - kernel) / stride + 1)
    w_out = math.floor((w + 2 * padding - kernel) / stride + 1)
    return h_out, w_out

def resnet50_real_weight_layers(input_h, input_w):
    h, w = input_h, input_w
    layers = []

    # conv1
    h, w = compute_conv2d_output(h, w, 7, 2, 3)
    layers.append(("conv1", h, w))

    # === layer1 === (3 bottleneck blocks)
    for block in range(3):
        stride = 1
        if block == 0:
            stride = 1
            layers.append(("layer1.0.downsample.0 (1x1)", h, w))  # projection shortcut

        h1, w1 = compute_conv2d_output(h, w, 1, stride, 0)
        layers.append((f"layer1.{block}.conv1 (1x1)", h1, w1))

        h2, w2 = compute_conv2d_output(h1, w1, 3, 1, 1)
        layers.append((f"layer1.{block}.conv2 (3x3)", h2, w2))

        h, w = compute_conv2d_output(h2, w2, 1, 1, 0)
        layers.append((f"layer1.{block}.conv3 (1x1)", h, w))

    # === layer2 === (4 bottleneck blocks)
    for block in range(4):
        stride = 1 if block > 0 else 2
        if block == 0:
            layers.append(("layer2.0.downsample.0 (1x1)", h, w))  # projection shortcut

        h1, w1 = compute_conv2d_output(h, w, 1, stride, 0)
        layers.append((f"layer2.{block}.conv1 (1x1)", h1, w1))

        h2, w2 = compute_conv2d_output(h1, w1, 3, 1, 1)
        layers.append((f"layer2.{block}.conv2 (3x3)", h2, w2))

        h, w = compute_conv2d_output(h2, w2, 1, 1, 0)
        layers.append((f"layer2.{block}.conv3 (1x1)", h, w))

    # === layer3 === (6 bottleneck blocks)
    for block in range(6):
        stride = 1 if block > 0 else 2
        if block == 0:
            layers.append(("layer3.0.downsample.0 (1x1)", h, w))  # projection shortcut

        h1, w1 = compute_conv2d_output(h, w, 1, stride, 0)
        layers.append((f"layer3.{block}.conv1 (1x1)", h1, w1))

        h2, w2 = compute_conv2d_output(h1, w1, 3, 1, 1)
        layers.append((f"layer3.{block}.conv2 (3x3)", h2, w2))

        h, w = compute_conv2d_output(h2, w2, 1, 1, 0)
        layers.append((f"layer3.{block}.conv3 (1x1)", h, w))

    # === layer4 === (3 bottleneck blocks)
    for block in range(3):
        stride = 1 if block > 0 else 2
        if block == 0:
            layers.append(("layer4.0.downsample.0 (1x1)", h, w))  # projection shortcut

        h1, w1 = compute_conv2d_output(h, w, 1, stride, 0)
        layers.append((f"layer4.{block}.conv1 (1x1)", h1, w1))

        h2, w2 = compute_conv2d_output(h1, w1, 3, 1, 1)
        layers.append((f"layer4.{block}.conv2 (3x3)", h2, w2))

        h, w = compute_conv2d_output(h2, w2, 1, 1, 0)
        layers.append((f"layer4.{block}.conv3 (1x1)", h, w))

    # fc (after avgpool)
    h, w = 1, 1
    layers.append(("fc (linear)", h, w))

    return layers
# ========================================================================

# ========== Load baseline ResNet50 ==========
baseline_model = tf.keras.models.load_model("Hetero-PIM/baseline_resnet50_tf_75p8_top1acc.h5")

NUM_LAYERS = 54 # should be 54 with shortcut layers

# get layer sizes (only Conv2D and Dense)
# ================================== Sohan =====================================================
# ******** I tried to adopt the codes needed for pytorch model to tf.
# ******** Could not run it since I dont have tf model. Please check if it works.
# ******** basically this part needs to save only the (weight shape, conv/linear). Ignore bias.  
# ==============================================================================================
layer_sizes = []
for layer in baseline_model.layers:
    try:
        params = np.sum([np.prod(w.shape) for w in layer.weights])
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_sizes.append((layer.weight.shape,"conv"))
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_sizes.append((layer.weight.shape,"linear"))
    except:
        continue

# ============ Need output shape to calculate compute latency ==============
layer_output_shapes = resnet50_real_weight_layers(224, 224)
# ==========================================================================

# ============ Load Timeloop simulated layer energies-Sohan ================
with open('resnet50_layer_energy_uJ.json', 'r') as f:
    resnet50_layer_energy_uJ = json.load(f)
# ==========================================================================

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
        # ============= Modified area,latency,energy calculation-Sohan =================
        for i in range(NUM_LAYERS):
            mem = "ReRAM" if i < 42 else "SRAM"
            memprops = MEMORY_TYPES[mem]
            
            # =========== Change start from here =======================================
            # Recollect necessary params
            layer_type = layer_sizes[i][-1]
            layer_shape = layer_sizes[i]
            _, layer_p, layer_q = layer_output_shapes[i]
            
            # Calculate area
            # ********** please verify the following comments related to tensorflow
            if layer_type == "conv":
                layer_rows = math.prod(layer_shape[0][:-1]) # [0][:-1] for tensorflow, [0][1:] for torch
                layer_cols = layer_shape[0][-1] # [0][-1] for tensorflow, [0][0] for torch
            else:
                layer_rows = layer_shape[0][0] # [0][0] for tensorflow, [0][1] for torch
                layer_cols = layer_shape[0][1] # [0][1] for tensorflow, [0][0] for torch
            layer_tiles = get_occupied_tiles(layer_rows,layer_cols)
            ## compute area
            layer_area = get_area(layer_tiles, mem)
            area += layer_area
            
            # Calculate latency
            ## compute necessary values
            mvms = layer_p*layer_q
            seperate_writes = layer_cols*(datawidth/bits_per_cell)
            device_idx = devices.index(mem)
            ## compute latency values
            compute_latency = (((1 * cycles_per_mvm)+(mvms-1)) * cycle_time) #with pipelining
            write_latency = write_latencies[device_idx] * seperate_writes
            total_latency = compute_latency+write_latency
            latency += total_latency
            
            # Calculate energy
            layer_energy = resnet50_layer_energy_uJ[str(i)]
            layer_energy = sum(layer_energy.values())
            energy += layer_energy
            # ====================================================================
            
            # ******** I could not do lifetime as it is more complicated
            # ******** lifetime cannot be calculated based on individual layer estimations
            # ******** Need the overall architecture.
            # ******** Like how many layers in writable crossbars?
            # ******** Can the writable crossbar store all of the rest of layers or partially, etc.

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
