import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# === CONFIG ===
RESNET_CHECKPOINT = "Hetero-PIM/baseline_resnet50_tf_75p8_top1acc.h5"
EXIT_LAYER = 33  # set your desired exit layer
NUM_CLASSES = 1000  # ImageNet

# === Load ResNet50 ===
baseline_model = tf.keras.models.load_model(RESNET_CHECKPOINT)

# get the output of the chosen exit layer
exit_layer_name = baseline_model.layers[EXIT_LAYER].name
exit_layer_output = baseline_model.get_layer(exit_layer_name).output

# === Attach branch classifier ===
branch_head = layers.GlobalAveragePooling2D()(exit_layer_output)
branch_head = layers.Dense(512, activation="relu")(branch_head)
branch_head = layers.Dense(NUM_CLASSES, activation="softmax")(branch_head)

early_exit_model = Model(inputs=baseline_model.input, outputs=branch_head)

# === Freeze up to the exit layer ===
for layer in early_exit_model.layers[:EXIT_LAYER]:
    layer.trainable = False

# === Compile ===
early_exit_model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === Placeholder data loader ===
# replace this with a proper ImageNet tf.data pipeline
# this is just a dummy for demonstration:
X_train = np.random.randn(100, 224, 224, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)

# === Train the branch classifier ===
print(f"\n[Training branch classifier at exit layer {EXIT_LAYER}]")
early_exit_model.fit(X_train, y_train, epochs=50, batch_size=8)

# === Evaluate ===
scores = early_exit_model.evaluate(X_train, y_train)
branch_acc = scores[1] * 100

print(f"\nMeasured Early Exit Accuracy at Layer {EXIT_LAYER}: {branch_acc:.2f}%")
