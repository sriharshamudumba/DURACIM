import numpy as np
import tensorflow as tf

def estimate_reram_layer_fit(
    model_path,
    reram_capacity_MB=16,
    bitwidth=8,
    crossbar_size=128
):
    """
    Estimate how many logical ResNet50 layers
    (49 Conv2D + 1 Dense) fit in given ReRAM capacity.
    """

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Only count Conv2D and Dense layers
    layer_sizes = []
    layer_names = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            params = np.sum([np.prod(w.shape) for w in layer.weights])
            layer_sizes.append(params)
            layer_names.append(layer.name)

    # convert bitwidth to bytes
    bytes_per_weight = bitwidth / 8
    total_capacity_bytes = reram_capacity_MB * 1024 * 1024

    cumulative = 0
    layers_on_reram = []

    for i, weight_count in enumerate(layer_sizes):
        this_size_bytes = weight_count * bytes_per_weight
        if cumulative + this_size_bytes <= total_capacity_bytes:
            cumulative += this_size_bytes
            layers_on_reram.append((i, layer_names[i]))
        else:
            break

    print("\n========= ReRAM Layer Mapping Summary =========")
    print(f"Total ReRAM capacity: {reram_capacity_MB} MB")
    print(f"Quantization: {bitwidth} bits per weight")
    print(f"Crossbar size assumed: {crossbar_size}")
    print(f"Total ResNet50 layers that fit: {len(layers_on_reram)} out of 50")
    print("Layers indices and names:")
    for idx, name in layers_on_reram:
        print(f"  - {idx}: {name}")
    print(f"Total used capacity: {cumulative/1024/1024:.2f} MB")
    print("================================================")

    return layers_on_reram


if __name__ == "__main__":
    reram_layers = estimate_reram_layer_fit(
        model_path="Hetero-PIM/baseline_resnet50_tf_75p8_top1acc.h5",
        reram_capacity_MB=16,
        bitwidth=16,
        crossbar_size=128
    )
