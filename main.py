import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import itertools


# Dataset Configuration
class LogicDataset(Dataset):
    def __init__(self, root_dir, feature_cols=["A", "B", "C"], label_col="D"):
        try:
            self.df = pd.read_csv(root_dir + "/logic_dataset.csv")
        except:
            return

        self.features = torch.tensor(
            self.df[feature_cols].values.astype(np.float32), dtype=torch.float32
        )

        label_mapping = {"True": 1, "False": 0, True: 1, False: 0}
        self.labels = torch.tensor(
            self.df[label_col].map(label_mapping).values.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Deterministic Architecture
class ExpansionLayer(nn.Module):
    def __init__(self, input_dim=3, max_coeff=1):
        super(ExpansionLayer, self).__init__()

        # Generate all combinations of coefficients [-1, 0, 1]
        coeffs = list(range(-max_coeff, max_coeff + 1))

        # Create a grid of all possible weight vectors
        # e.g. [1, 1, 0] represents (1*A + 1*B + 0*C) -> (A+B)
        combinations = list(itertools.product(coeffs, repeat=input_dim))

        # Convert to tensor [Num_Combos, Input_Dim]
        self.weights = torch.tensor(combinations, dtype=torch.float32).T
        self.weights = nn.Parameter(self.weights, requires_grad=False)

        self.out_dim = self.weights.shape[1]

    def forward(self, x):
        # Matrix multiplication: Batch x Combos
        return x @ self.weights


class DeterministicNet(nn.Module):
    def __init__(self):
        super().__init__()

        # generate combinations of A, B, C using coefficients [-1, 0, 1]
        self.logic_basis = ExpansionLayer(input_dim=3, max_coeff=3)

        input_dim = (
            3 + self.logic_basis.out_dim
        )  # raw inputs + parity features (3 + 343 = 346 features)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # calculate all integer combinations (odd/even sums)
        basis_sums = self.logic_basis(x)

        # convert to cosine features for better separability (-1 to 1 range)
        cos_feats = torch.cos(np.pi * basis_sums)

        combined = torch.cat([x, cos_feats], dim=-1)

        return self.net(combined)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LogicDataset(root_dir="./data")

    labels = dataset.labels.numpy()
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=len(test_idx), shuffle=False
    )

    print("Initializing Deterministic Model...")
    model = DeterministicNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training on {len(train_idx)} | Testing on {len(test_idx)}")
    print("-" * 50)
    print(f"| {'Epoch':^5} | {'Train Acc':^10} | {'Test Acc':^10} |")
    print("-" * 50)

    for epoch in range(21):
        model.train()
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = 100 * correct / total

        # Evaluate
        model.eval()
        failures = []
        successes = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                for i in range(len(targets)):
                    sample_data = {
                        "input": inputs[i].cpu().numpy(),
                        "predicted": predicted[i].item(),
                        "true": targets[i].item(),
                        "probability": probs[i].item(),
                        "confidence": abs(probs[i].item() - 0.5),
                    }

                    if predicted[i] == targets[i]:
                        successes.append(sample_data)
                    else:
                        failures.append(sample_data)

        test_acc = 100 * len(successes) / (len(successes) + len(failures))

        if epoch % 10 == 0:
            print(f"| {epoch:5d} | {train_acc:9.2f}% | {test_acc:9.2f}% |")

    print("\nTesting with new test set:")
    model.eval()

    x1 = torch.tensor([[-24, 3, 6]], dtype=torch.float32).to(device)
    x2 = torch.tensor([[10, 3, 4]], dtype=torch.float32).to(device)
    x3 = torch.tensor([[-20, 3, 13]], dtype=torch.float32).to(device)
    x4 = torch.tensor([[-7, 15, -8]], dtype=torch.float32).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(model(x1)).item()
        p2 = torch.sigmoid(model(x2)).item()
        p3 = torch.sigmoid(model(x3)).item()
        p4 = torch.sigmoid(model(x4)).item()

    print(
        f"1. [-24, 3, 6]  (Expect False): {p1:.5f} -> {'PASS' if p1 < 0.5 else 'FAIL'}"
    )
    print(
        f"2. [ 10, 3, 4]  (Expect True) : {p2:.5f} -> {'PASS' if p2 > 0.5 else 'FAIL'}"
    )
    print(
        f"3. [-20, 3, 13] (Expect True) : {p3:.5f} -> {'PASS' if p3 > 0.5 else 'FAIL'}"
    )
    print(
        f"3. [-7, 15, 8] (Expect False) : {p4:.5f} -> {'PASS' if p4 < 0.5 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
