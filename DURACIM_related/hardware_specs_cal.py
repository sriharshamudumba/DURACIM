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

# ========== Baseline metrics ==========
baseline_latency = 0
baseline_area = 0
baseline_energy = 0
baseline_lifetimes = []
for i in range(NUM_LAYERS):
    memprops = MEMORY_TYPES["SRAM"]
    n_crossbars = 1
    try:
        shape = layer_sizes[i][0]
        if isinstance(shape, tuple):
            rows = math.prod(shape[:-1])
            cols = shape[-1]
            n_crossbars = math.ceil((rows * cols) / CROSSBAR_SIZE)
    except:
        continue
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
