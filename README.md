# Hybrid Quantum-Classical Chest X-Ray Classification

An experimental hybrid quantum-classical machine learning pipeline for multi-class chest X-ray classification using an **8-qubit Variational Quantum Circuit (VQC)**.

Developed for **IBM Qiskit Fall Fest 2026 — Track 08: Quantum AI for Healthcare & Biomedical Diagnostics**.

> **Important:** This project is a research/hackathon prototype. It is **not a clinical diagnostic system** and has not been clinically validated.

---

## Overview

This project explores how a quantum machine-learning component can be integrated into a medical image-classification pipeline.

Instead of directly processing high-dimensional X-ray images with a quantum circuit, we use a hybrid approach:

```text
Chest X-Ray
     ↓
Pretrained ResNet-18
     ↓
512-D Feature Representation
     ↓
StandardScaler + PCA
     ↓
8-D Quantum Representation
     ↓
8-Qubit Variational Quantum Circuit
     ↓
4-Class Prediction
```

The four classification classes are:

* COVID-19
* Normal
* Pneumonia-Bacterial
* Pneumonia-Viral

---

## Quantum Architecture

The final quantum classifier uses:

* **8 qubits**
* **2 trainable layers**
* **32 trainable parameters**
* Initial `Ry` data encoding
* Trainable `Ry` and `Rz` rotations
* CX entanglement ladder
* Data re-uploading using `Ry`
* Qiskit `SamplerQNN`
* Measurement-based 4-class interpretation

The quantum circuit was implemented using **Qiskit Machine Learning**.

---

## Data & Preprocessing

A curated chest X-ray dataset was used for the experiments.

Before the final experiments, the dataset was checked for exact duplicate images using file hashing. One same-class duplicate pair was identified and excluded from the working split.

The final split contained:

| Split      |   Samples |
| ---------- | --------: |
| Training   |     6,445 |
| Validation |     1,381 |
| Test       |     1,382 |
| **Total**  | **9,208** |

The image pipeline was:

1. Resize and preprocess X-rays using the pretrained ResNet-18 transformation.
2. Extract a 512-dimensional feature representation.
3. Standardize the extracted features.
4. Apply PCA.
5. Retain 8 principal components for the quantum model.
6. Scale the resulting quantum inputs to the range `[-π, π]`.

The 8-dimensional representation was selected as a practical **representation-vs-qubit-cost trade-off**, rather than because it produced the best classical performance.

---

## Experimental Results

### Final VQC

The frozen final VQC achieved:

| Metric            | Test Result |
| ----------------- | ----------: |
| Accuracy          |  **34.95%** |
| Balanced Accuracy |  **31.03%** |
| Macro-F1          |  **30.58%** |

### Fair Classical Baseline

For a fair comparison, a classical model was trained using the same balanced quantum-training budget of **200 samples (50 per class)**.

| Model              |   Accuracy | Balanced Accuracy |   Macro-F1 |
| ------------------ | ---------: | ----------------: | ---------: |
| Classical baseline | **73.73%** |        **74.00%** | **72.52%** |
| 8-Qubit VQC        | **34.95%** |        **31.03%** | **30.58%** |

### Important Finding

**No quantum advantage was observed in this experiment.**

The classical baseline substantially outperformed the VQC under the tested configuration.

Rather than presenting the result as a quantum advantage, this project reports the actual experimental outcome and investigates the limitations of the current approach.

---

## Noise Experiments

The quantum circuit was also evaluated under simulated noise.

Two noise models were tested:

* Depolarizing noise
* Thermal relaxation noise

The tested 100-sample balanced subset produced approximately:

| Configuration      | Accuracy |
| ------------------ | -------: |
| Ideal              |      33% |
| Depolarizing noise |      33% |
| Thermal noise      |      33% |

The results did not show a substantial change under the tested noise configuration.

### Zero-Noise Extrapolation

Zero-Noise Extrapolation (ZNE) was also evaluated.

The tested ZNE configuration produced:

**28% accuracy**

and therefore did not improve
