import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\joydrot banerjee\OneDrive\Desktop\Chest_Xray_QML"

TEST_CSV = os.path.join(ROOT, "splits", "test.csv")

TRAIN_PCA = os.path.join(
    ROOT, "pca_features", "pca8", "train_pca.npy"
)

TEST_PCA = os.path.join(
    ROOT, "pca_features", "pca8", "test_pca.npy"
)

TRAIN_LABELS = os.path.join(
    ROOT, "features", "train_labels.npy"
)

TEST_LABELS = os.path.join(
    ROOT, "features", "test_labels.npy"
)

WEIGHTS_PATH = os.path.join(
    ROOT, "quantum_results", "cp09b_direct_trained_weights.npy"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "COVID-19",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral"
]

N_CLASSES = 4
N_QUBITS = 8
N_LAYERS = 2


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("HYBRID QUANTUM-CLASSICAL CHEST X-RAY DEMO")
print("=" * 65)

df_test = pd.read_csv(TEST_CSV)

X_train = np.load(TRAIN_PCA)
X_test = np.load(TEST_PCA)

y_train = np.load(TRAIN_LABELS)
y_test = np.load(TEST_LABELS)

weights = np.load(WEIGHTS_PATH)

print("\nDataset:")
print("Training features :", X_train.shape)
print("Test features     :", X_test.shape)
print("Frozen weights    :", weights.shape)


# ============================================================
# VERIFY TEST SAMPLE 0
# ============================================================

sample_index = 0

row = df_test.iloc[sample_index]

image_path = row["filepath"]
csv_class = row["class"]
csv_label = int(row["label"])

numpy_label = int(y_test[sample_index])

print("\nTest sample verification:")
print("CSV class         :", csv_class)
print("CSV label         :", csv_label)
print("NumPy label       :", numpy_label)
print("Image             :", image_path)

if csv_label != numpy_label:
    raise ValueError(
        "CSV label and NumPy label do not match."
    )

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"X-ray image not found:\n{image_path}"
    )


# ============================================================
# REPRODUCE CP09-B BALANCED TRAINING SUBSET
# ============================================================

rng = np.random.default_rng(42)

selected_indices = []

for class_id in range(N_CLASSES):

    class_indices = np.where(y_train == class_id)[0]

    chosen = rng.choice(
        class_indices,
        size=50,
        replace=False
    )

    selected_indices.extend(chosen)

selected_indices = np.array(selected_indices)

X_train_balanced = X_train[selected_indices]
y_train_balanced = y_train[selected_indices]

print("\nBalanced training subset:")
print("Samples:", len(X_train_balanced))

print(
    "Class distribution:",
    np.bincount(y_train_balanced, minlength=N_CLASSES)
)


# ============================================================
# REPRODUCE CP09-B MIN-MAX SCALING
# ============================================================

train_min = X_train_balanced.min(axis=0)
train_max = X_train_balanced.max(axis=0)

denominator = train_max - train_min

denominator[denominator == 0] = 1.0

X_test_scaled = (
    (X_test - train_min)
    / denominator
)

X_test_scaled = (
    X_test_scaled * (2 * np.pi)
    - np.pi
)

x_sample = X_test_scaled[sample_index]


# ============================================================
# BUILD THE EXACT FROZEN VQC ARCHITECTURE
# ============================================================

x = ParameterVector("x", N_QUBITS)
theta = ParameterVector("theta", 32)

qc = QuantumCircuit(N_QUBITS)

# Initial data encoding
for i in range(N_QUBITS):
    qc.ry(x[i], i)

theta_index = 0

for layer in range(N_LAYERS):

    # Trainable Ry/Rz rotations
    for qubit in range(N_QUBITS):

        qc.ry(theta[theta_index], qubit)
        theta_index += 1

        qc.rz(theta[theta_index], qubit)
        theta_index += 1

    # CX ladder
    for qubit in range(N_QUBITS - 1):
        qc.cx(qubit, qubit + 1)

    # Data re-uploading
    for qubit in range(N_QUBITS):
        qc.ry(x[qubit], qubit)


# ============================================================
# MEASUREMENT
# ============================================================

qc.measure_all()


def interpret_state(state):
    return state % 4


qnn = SamplerQNN(
    circuit=qc,
    input_params=x,
    weight_params=theta,
    interpret=interpret_state,
    output_shape=N_CLASSES
)


# ============================================================
# RUN THE FROZEN VQC
# ============================================================

print("\nQuantum circuit:")
print(qc)

print("\nCircuit information:")
print("Qubits              :", N_QUBITS)
print("Trainable parameters:", len(theta))
print("Layers              :", N_LAYERS)
print("Circuit depth       :", qc.depth())
print("Gate count          :", len(qc.data))


print("\nRunning frozen VQC on test sample 0...")

probabilities = qnn.forward(
    x_sample.reshape(1, -1),
    weights
)[0]

probabilities = np.asarray(probabilities, dtype=float)

predicted_label = int(np.argmax(probabilities))
true_label = int(y_test[sample_index])


# ============================================================
# PRINT RESULT
# ============================================================

print("\n" + "=" * 65)
print("VQC PREDICTION")
print("=" * 65)

print("\nTrue class     :", CLASS_NAMES[true_label])
print("Predicted class:", CLASS_NAMES[predicted_label])

print("\nClass probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    probabilities
):
    print(
        f"{class_name:<22}: "
        f"{probability:.4f} "
        f"({probability * 100:.2f}%)"
    )


# ============================================================
# FROZEN OVERALL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FROZEN EXPERIMENT RESULTS")
print("=" * 65)

print("\nFair classical baseline:")
print("Accuracy     : 73.73%")
print("Balanced Acc : 74.00%")
print("Macro F1     : 72.52%")

print("\nVQC:")
print("Accuracy     : 34.95%")
print("Balanced Acc : 31.03%")
print("Macro F1     : 30.58%")

print("\nNoise experiments:")
print("Ideal        : ~33% accuracy")
print("Depolarizing : ~33% accuracy")
print("Thermal      : ~33% accuracy")
print("ZNE          : ~28% accuracy")


# ============================================================
# DISPLAY ACTUAL X-RAY
# ============================================================

image = Image.open(image_path).convert("RGB")

plt.figure(figsize=(8, 7))

plt.imshow(image)
plt.axis("off")

plt.title(
    f"Test Sample {sample_index}\n"
    f"True: {CLASS_NAMES[true_label]} | "
    f"VQC Prediction: {CLASS_NAMES[predicted_label]}"
)

plt.tight_layout()
plt.show()


# ============================================================
# DISPLAY PROBABILITIES
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    CLASS_NAMES,
    probabilities
)

plt.ylabel("Probability")
plt.xlabel("Class")
plt.title("Frozen VQC Output Probabilities")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()
plt.show()

print("\nDemo completed successfully.")