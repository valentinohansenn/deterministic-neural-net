# Logic Prediction Neural Network

A PyTorch implementation that predicts binary logic outcomes from three numerical inputs using deterministic feature expansion and cosine encoding.

## Overview

This project uses a neural network to learn complex logical relationships between three input variables (A, B, C) and predict a binary outcome (D: True/False). The model employs feature engineering to explicitly generate all possible linear combinations of inputs, transformed through cosine encoding for better separability.

## Architecture

**ExpansionLayer**
- Generates 343 deterministic features using coefficients {-3, -2, -1, 0, 1, 2, 3}
- Creates all combinations: `coeff_a * A + coeff_b * B + coeff_c * C`
- Non-trainable (fixed weights)

**DeterministicNet**
- Input: 346 features (3 original + 343 combinations transformed with cosine)
- Hidden layers: 346 → 128 → 64 → 1
- Activation: GELU
- Output: Binary classification with sigmoid

## Dataset

The dataset (`data/logic_dataset.csv`) contains:
- **Features**: A, B, C (numerical values)
- **Label**: D (True/False)
- **Split**: 80% train, 20% test (stratified)

## Requirements

```
torch>=2.9.1
pandas>=2.3.3
numpy>=2.3.5
scikit-learn>=1.7.2
matplotlib>=3.10.7
tqdm>=4.67.1
```

## Installation

Using pip:
```bash
pip install -r requirements.txt
```

Using uv (recommended):
```bash
uv sync
```

## Usage

```bash
python main.py
```

### Output

The script will:
1. Train the model for 20 epochs
2. Display training and test accuracy every 10 epochs
3. Test on custom samples with predictions and confidence scores

Example output:
```
Training on 800 | Testing on 200
--------------------------------------------------
| Epoch | Train Acc  | Test Acc   |
--------------------------------------------------
|     0 |    85.50%  |    87.00%  |
|    10 |    99.25%  |    98.50%  |
|    20 |    99.88%  |    99.00%  |

Testing with new test set:
1. [-24, 3, 6]  (Expect False): 0.12345 -> PASS
2. [ 10, 3, 4]  (Expect True) : 0.89234 -> PASS
3. [-20, 3, 13] (Expect True) : 0.91567 -> PASS
4. [-7, 15, -8] (Expect False): 0.08923 -> PASS
```

## How It Works

1. **Feature Expansion**: Generate all linear combinations of inputs with integer coefficients
2. **Cosine Transform**: Apply `cos(π * sum)` to capture parity patterns
3. **Neural Network**: Feed expanded features through a 3-layer network
4. **Binary Classification**: Output probabilities for True/False prediction

## Key Design Choices

- **Deterministic expansion** instead of learned embeddings ensures coverage of all linear relationships
- **Cosine encoding** maps integer sums to periodic features, making odd/even patterns more learnable
- **Fixed expansion weights** reduce overfitting while maintaining expressiveness
- **Stratified split** ensures balanced class distribution in train/test sets

## Project Structure

```
v-logic-prediction/
├── data/
│   ├── logic_dataset.csv          # Full dataset
│   └── logic_dataset_sample.csv   # Sample data
├── main.py                         # Training script
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## License

MIT
