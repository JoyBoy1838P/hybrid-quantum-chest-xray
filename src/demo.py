"""
IBM Qiskit Fall Fest 2026
Track 8 — Quantum AI for Healthcare & Biomedical Diagnostics

FINAL DEMO SCRIPT

Purpose:
    Demonstrate the frozen CP09-B VQC experiment.

Important:
    - No training
    - No parameter updates
    - No dataset modification
    - No experiment modification
    - Uses the frozen CP09-B weights
"""

from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PCA_DIR = PROJECT_DIR / "pca_features" / "pca8"
FEATURE_DIR = PROJECT_DIR / "features"
RESULTS_DIR = PROJECT_DIR / "quantum_results"


# ============================================================
# 2. CONFIGURATION
# ============================================================

N_QUBITS = 8
N_CLASSES = 4
N_LAYERS = 2

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 50

CLASS_NAMES = [
    "COVID-19",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral"
]


# ============================================================
# 3. LOAD FROZEN DATA
# ============================================================

print("=" * 70)
print("QISKIT FALL FEST 2026 — FINAL VQC DEMO")
print("=" * 70)

print("\n[1/7] Loading frozen project data...")

X_train = np.load(PCA_DIR / "train_pca.npy")
y_train = np.load(FEATURE_DIR / "train_labels.npy")

X_test = np.load(PCA_DIR / "test_pca.npy")
y_test = np.load(FEATURE_DIR / "test_labels.npy")

weights = np.load(
    RESULTS_DIR / "cp09b_direct_trained_weights.npy"
)

print(f"Training features : {X_train.shape}")
print(f"Test features     : {X_test.shape}")
print(f"Frozen weights    : {weights.shape}")


# ============================================================
# 4. REPRODUCE CP09-B TRAINING SUBSET
# ============================================================

print("\n[2/7] Reproducing CP09-B feature scaling...")

rng = np.random.default_rng(RANDOM_SEED)

selected_indices = []

for class_id in range(N_CLASSES):

    class_indices = np.where(y_train == class_id)[0]

    chosen = rng.choice(
        class_indices,
        size=SAMPLES_PER_CLASS,
        replace=False
    )

    selected_indices.extend(chosen)


selected_indices = np.array(selected_indices)

X_train_balanced = X_train[selected_indices]
y_train_balanced = y_train[selected_indices]

print(
    f"Balanced training subset: "
    f"{len(X_train_balanced)} samples"
)

print(
    "Class distribution:",
    np.bincount(y_train_balanced)
)


# ============================================================
# 5. REPRODUCE CP09-B [-PI, PI] SCALING
# ============================================================

train_min = X_train_balanced.min(axis=0)
train_max = X_train_balanced.max(axis=0)

range_values = train_max - train_min

# Protect against division by zero.
range_values[range_values == 0] = 1.0


def scale_to_pi(X):
    """
    Reproduce the CP09-B min-max scaling:

        x_scaled = 2π * normalized_x - π

    Scaling parameters come ONLY from the
    frozen balanced training subset.
    """

    normalized = (
        (X - train_min) /
        range_values
    )

    return (
        2.0 * np.pi * normalized
    ) - np.pi


X_test_scaled = scale_to_pi(X_test)


# ============================================================
# 6. BUILD THE EXACT CP09-B CIRCUIT
# ============================================================

print("\n[3/7] Building frozen 8-qubit VQC...")

x = ParameterVector("x", N_QUBITS)
theta = ParameterVector(
    "theta",
    N_QUBITS * 2 * N_LAYERS
)

qc = QuantumCircuit(N_QUBITS)


# ------------------------------------------------------------
# Initial angle encoding
# ------------------------------------------------------------

for q in range(N_QUBITS):
    qc.ry(x[q], q)


# ------------------------------------------------------------
# Trainable layers + CX ladder + data re-uploading
# ------------------------------------------------------------

theta_index = 0

for layer in range(N_LAYERS):

    # Trainable Ry/Rz rotations
    for q in range(N_QUBITS):

        qc.ry(
            theta[theta_index],
            q
        )
        theta_index += 1

        qc.rz(
            theta[theta_index],
            q
        )
        theta_index += 1

    # CX ladder
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)

    # Data re-uploading
    for q in range(N_QUBITS):
        qc.ry(x[q], q)


# Measurement
qc.measure_all()


print(f"Qubits              : {N_QUBITS}")
print(f"Trainable parameters: {len(theta)}")
print(f"Layers              : {N_LAYERS}")
print(f"Circuit depth       : {qc.depth()}")
print(f"Gate count          : {len(qc.data)}")


# ============================================================
# 7. DEFINE CLASS INTERPRETATION
# ============================================================

def interpret_state(state):
    """
    Map measured computational-basis states
    into the four output classes.
    """

    return state % N_CLASSES


# ============================================================
# 8. CREATE SAMPLERQNN
# ============================================================

print("\n[4/7] Creating Qiskit SamplerQNN...")

qnn = SamplerQNN(
    circuit=qc,
    input_params=list(x),
    weight_params=list(theta),
    interpret=interpret_state,
    output_shape=N_CLASSES
)

print("SamplerQNN created successfully.")


# ============================================================
# 9. DISPLAY CIRCUIT
# ============================================================

print("\n[5/7] FINAL QUANTUM CIRCUIT")
print("-" * 70)

print(qc.draw(output="text"))


# ============================================================
# 10. RUN ONE FIXED TEST SAMPLE
# ============================================================

DEMO_SAMPLE_INDEX = 0

sample = X_test_scaled[
    DEMO_SAMPLE_INDEX:
    DEMO_SAMPLE_INDEX + 1
]

true_label = int(
    y_test[DEMO_SAMPLE_INDEX]
)


print("\n[6/7] Running frozen VQC on test sample...")

probabilities = qnn.forward(
    sample,
    weights
)

probabilities = np.asarray(
    probabilities
).reshape(-1)

predicted_label = int(
    np.argmax(probabilities)
)


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VQC DEMO RESULT")
print("=" * 70)

print(f"\nTest sample index : {DEMO_SAMPLE_INDEX}")

print(
    f"True class        : "
    f"{CLASS_NAMES[true_label]}"
)

print(
    f"Predicted class   : "
    f"{CLASS_NAMES[predicted_label]}"
)

print("\nClass probabilities:")

for class_id, probability in enumerate(probabilities):

    print(
        f"  {CLASS_NAMES[class_id]:20s} "
        f"{probability:.4f}"
    )


# ============================================================
# 12. FROZEN FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FROZEN PROJECT RESULTS")
print("=" * 70)

print("\nMatched 200-sample comparison:")

print(
    "Classical Logistic Regression"
    " : 73.73% accuracy"
)

print(
    "8-qubit VQC"
    "                    : 34.95% accuracy"
)

print("\nNoise experiments:")

print("Ideal              : 33%")
print("Depolarizing       : 33%")
print("Thermal relaxation : 33%")
print("ZNE                : 28%")


# ============================================================
# 13. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    "\nThe frozen VQC experiment did not demonstrate "
    "quantum advantage."
)

print(
    "The matched classical baseline substantially "
    "outperformed the VQC."
)

print(
    "This experiment demonstrates an end-to-end "
    "Qiskit QML pipeline and provides an honest "
    "measurement of its limitations."
)

print("\nDemo completed.")