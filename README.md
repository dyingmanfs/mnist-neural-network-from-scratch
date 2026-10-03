# CNG403 Assignment 1: Feed-Forward Neural Network From Scratch

## Overview

This assignment guides you through implementing a complete Feed-Forward Neural Network (FFNN) **from scratch** using NumPy and PyTorch, trained on the MNIST digit classification task using HOG (Histogram of Oriented Gradients) features.

**Key Learning Objectives:**
- Understand the mechanics of forward and backward propagation
- Implement core neural network components (layers, activations, loss functions)
- Assemble components into a complete training pipeline
- Compare your implementation against a PyTorch reference

---

## Project Structure

```
assignment1/
├── src/                          # Source code directory
│   ├── layers.py                 # [TODO] Implement: Linear, ReLU, Sigmoid, Tanh, CrossEntropyLoss
│   ├── network.py                # [TODO] Implement: FFNN network assembly and backprop
│   ├── optimizers.py             # [TODO] Implement: SGD optimizer with momentum
│   ├── train.py                  # Complete: Training pipeline (uses your implementations)
│   ├── reference.py              # Complete: PyTorch reference implementation
│   └── data_utils.py             # Complete: Shared data loading and HOG extraction
│
├── data/
│   └── MNIST/                    # Auto-populated with MNIST dataset on first run
│
├── checkpoints/
│   ├── reference_best.pt         # Saved weights from reference model
│   └── scratch_best.pt           # Your best model (created during training)
│
├── logs/
│   ├── reference_history.npy     # Training history from reference
│   └── scratch_history.npy       # Your training history
│
├── config.json                   # Configuration file (shared by reference and your implementation)
├── notebook.ipynb                # Main student interface — run 
└── README.md                     # This file
```

---

## Your Tasks: What to Implement

You need to implement **core components** in three files:

---

## Grading Policy

**Points are awarded ONLY if sanity checks pass.** Each TODO implementation must produce correct results.

| File | Components | Points | Sanity Check Required |
|------|-----------|--------|----------------------|
| `layers.py` | Linear.forward/backward, ReLU, Sigmoid, Tanh, CrossEntropyLoss | 40 pts | Must match PyTorch numerically |
| `network.py` | FFNN.__init__, forward, backward, l2_grad | 30 pts | Must match nn.Sequential |
| `optimizers.py` | SGD.step (vanilla + momentum) | 20 pts | Must converge correctly |
| `Hyperparameter Tuning` (Section 8) | Find best config, report test accuracy | 10 pts | Test set evaluation only |
| **Total** | | **100 pts** | |

**Important**: Your model must achieve **≥95% test accuracy** to pass. Reference achieves 97.14%.

### 1. **`src/layers.py`** — Neural Network Layers (3 TODOs)

Implement:
- **`Linear.forward(x)`**: Affine transformation \( y = xW^T + b \)
- **`Linear.backward(grad_out)`**: Backprop to compute gradients for weights and inputs
- **`ReLU.forward/backward()`**: ReLU activation
- **`Sigmoid.forward/backward()`**: Sigmoid activation
- **`Tanh.forward/backward()`**: Tanh activation
- **`CrossEntropyLoss.forward/backward()`**: Cross-entropy loss + softmax

**Validation:** Each layer has built-in sanity checks comparing against PyTorch (cell 11 in notebook).

---

### 2. **`src/network.py`** — Network Assembly (2 TODOs)

Implement:
- **`FFNN.__init__()`**: Assemble `Linear` and activation layers sequentially
- **`FFNN.forward(x)`**: Forward pass through all layers
- **`FFNN.backward(grad)`**: Backpropagation in reverse layer order
- **`FFNN.l2_grad(lambda)`**: Add L2 regularization gradient

**Validation:** Network-level sanity checks against `nn.Sequential` (cell 15 in notebook).

---

### 3. **`src/optimizers.py`** — Parameter Updates (1 TODO)

Implement:
- **`SGD.step()`**: Update rule: \( W \leftarrow W - \text{lr} \times \nabla W \)
- **`SGD.step()` with momentum**: \( v \leftarrow m \times v + (1-m) \times \nabla W \)

**Validation:** Sanity checks in the file and notebook (cell 15).

---

## What's Already Provided

- **`src/train.py`**: Complete training loop — reads your implementations and trains them
- **`src/reference.py`**: Full PyTorch reference (study this to understand expected behavior)
- **`src/data_utils.py`**: Shared utilities for:
  - MNIST loading and HOG feature extraction (144-dim features per image)
  - Data standardization and train/val/test splits
- **`configs/config.json`**: Unified configuration for both reference and your implementation
- **`notebook.ipynb`**: Interactive guide through all steps

---

## Getting Started

### Step 1: Review the Notebook Structure

Open `notebook.ipynb` and follow these 8 sections in order:

1. **Setup** — Import libraries, load config
2. **Data & Features** — Visualize MNIST → HOG conversion
3. **Reference Model** — Run PyTorch reference (establishes ground truth: ~97% test accuracy)
4. **Your Layers** — Implement `layers.py`, run sanity checks (cell 11)
5. **Your Network & Optimizer** — Implement `network.py` and `optimizers.py`, run sanity checks (cell 15)
6. **Training** — Run your training pipeline (cell 19)
7. **Comparison** — Plot learning curves: yours vs. reference
8. **Tuning** — (Optional) Adjust hyperparameters and re-train

### Step 2: Implement in order

**Don't skip around.** The sanity checks depend on earlier work:

```
✓ Implement layers.py → Run sanity checks (cell 11)
  ↓
✓ Implement network.py → Run sanity checks (cell 15)
  ↓
✓ Implement optimizers.py → Run sanity checks (cell 15)
  ↓
✓ Train your model (cell 19)
```

### Step 3: Validate Your Work

Your model should achieve **similar accuracy to the reference** (~97% test accuracy).

The notebook provides:
- Side-by-side training/validation curves
- Sanity checks that compare against PyTorch
- Performance comparison

---

## Data Details

- **Dataset**: MNIST (60K training, 10K test)
- **Features**: HOG (Histogram of Oriented Gradients)
  - 144-dimensional feature vectors per image
  - Fixed HOG parameters: 9 orientations, 8×8 pixels per cell, 2×2 cells per block
  - Standardized (zero-mean, unit-variance) using training set statistics
- **Train/Val/Test Split**: 90% train, 10% validation, 100% test
  - Train: 54,000 samples
  - Val: 6,000 samples
  - Test: 10,000 samples

---

## Configuration (`configs/config.json`)

All hyperparameters are in one file:

```json
{
  "training": {
    "batch_size": 64,           // Examples per batch
    "epochs": 20,               // Training epochs
    "learning_rate": 0.01,      // SGD learning rate
    "momentum": 0.9,            // SGD momentum
    "l2_lambda": 0.0001,        // L2 regularization coefficient
    "seed": 42                  // Random seed (reproducibility)
  },
  "model": {
    "hidden_sizes": [256, 128], // Two hidden layers
    "activation": "relu"        // Activation function
  },
  "paths": {
    "checkpoint_dir": "checkpoints/",
    "log_dir": "logs/"
  }
}
```

Network architecture (automatically built):
```
Input (144) → Linear(144→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10) → [Softmax in loss]
```

---

## Running the Code Directly (Optional)

After implementing, you can test directly from the terminal:

```bash
cd assignment1/src

# Run sanity checks for layers
python layers.py

# Run sanity checks for network + optimizers
python network.py

# Run full training
python train.py --config ../config.json

# Run reference (for comparison)
python reference.py --config ../config.json
```

---

## Expected Results

| Component | Expected Test Accuracy | Reference Test Accuracy |
|-----------|------------------------|-------------------------|
| Your FFNN | ~97% (within 1-2%)     | 97.14%                  |

**Note**: Small differences are normal due to:
- Differences in random initialization
- Gradient descent stochasticity (SGD)
- Floating-point precision

---

## File Reference: What Each File Does

| File | Purpose | Your Task? |
|------|---------|-----------|
| `layers.py` | ReLU, Linear, Loss | **YES** — Implement marked TODOs |
| `network.py` | Assemble layers into FFNN | **YES** — Implement marked TODOs |
| `optimizers.py` | SGD with momentum | **YES** — Implement marked TODOs |
| `train.py` | Training loop | No — Uses your implementations |
| `reference.py` | PyTorch reference | No — Study to understand behavior |
| `data_utils.py` | Data loading + HOG | No — Shared utility |
| `notebook.ipynb` | Interactive guide | Yes — Hyperparameter tuning |
| `config.json` | Hyperparameters | No — Modify to experiment |

---

## Troubleshooting

### "Module not found" errors
- Make sure you're in the `assignment1/src/` directory when running scripts
- The notebook handles imports automatically

### "MNIST download fails"
- Check internet connection (first run downloads MNIST ~60MB)
- Dataset will be cached in `assignment1/data/MNIST/`

### Your model trains but doesn't converge
- Check your backprop math (run cell 11 sanity checks first)
- Verify gradients match PyTorch reference in sanity checks
- Try lowering learning rate or increasing epochs in config

### Gradient mismatches in sanity checks
- Double-check matrix dimensions in forward/backward passes
- For backprop: ensure chain rule is applied correctly
- Verify activation derivatives are correct

---


Good luck! 
