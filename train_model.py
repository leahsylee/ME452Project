"""
Train an MLP to predict tool wear (Vbmax) from extracted features.

Reads processed CSV files from processed_data/train/ and processed_data/test/,
normalises features, trains a small fully-connected network, and reports metrics.
"""

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "processed_data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42


# ─── data loading ────────────────────────────────────────────────────────────
def load_data():
    """Load train/test CSVs and return normalised tensors + scaler params."""
    train_x = np.loadtxt(TRAIN_DIR / "train_features.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt(TRAIN_DIR / "train_labels.csv", delimiter=",", skiprows=1)
    test_x = np.loadtxt(TEST_DIR / "test_features.csv", delimiter=",", skiprows=1)
    test_y = np.loadtxt(TEST_DIR / "test_labels.csv", delimiter=",", skiprows=1)

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Log-transform columns with extreme dynamic range (e.g. spectral_energy)
    # before z-score normalization to prevent overflow
    for col in range(train_x.shape[1]):
        col_max = np.abs(train_x[:, col]).max()
        if col_max > 1e6:
            sign_tr = np.sign(train_x[:, col])
            sign_te = np.sign(test_x[:, col])
            train_x[:, col] = sign_tr * np.log1p(np.abs(train_x[:, col]))
            test_x[:, col] = sign_te * np.log1p(np.abs(test_x[:, col]))

    # z-score normalisation fitted on training set only
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0] = 1.0

    train_x = (train_x - mu) / sigma
    test_x = (test_x - mu) / sigma

    # Clip to prevent extreme outliers from causing NaN
    train_x = np.clip(train_x, -10, 10)
    test_x = np.clip(test_x, -10, 10)

    return (
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.float32),
    )


# ─── model ───────────────────────────────────────────────────────────────────
class ToolWearMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─── training loop ───────────────────────────────────────────────────────────
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * xb.size(0)
        preds.append(pred.cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()
    mse = total_loss / len(loader.dataset)
    rmse = np.sqrt(mse)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, rmse, r2, preds, targets


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading processed data...")
    train_x, train_y, test_x, test_y = load_data()
    print(f"  Train: {train_x.shape[0]} samples, {train_x.shape[1]} features")
    print(f"  Test:  {test_x.shape[0]} samples")

    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))

    model = ToolWearMLP(input_dim=train_x.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    num_epochs = 500
    train_losses, test_losses = [], []

    print(f"\nTraining for {num_epochs} epochs on {DEVICE}...")
    for epoch in range(1, num_epochs + 1):
        t_loss = train(model, train_loader, criterion, optimizer)
        _, _, _, _, _ = evaluate(model, train_loader, criterion)
        te_mse, te_rmse, te_r2, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(t_loss)
        test_losses.append(te_mse)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}  |  Train MSE: {t_loss:.6f}  |  "
                  f"Test RMSE: {te_rmse:.4f}  |  Test R²: {te_r2:.4f}")

    # ── final evaluation ──
    _, final_rmse, final_r2, preds, targets = evaluate(model, test_loader, criterion)

    print(f"\n{'='*60}")
    print(f"  {'Sample':>6}  {'Predicted':>10}  {'Actual':>10}  {'Accuracy':>10}")
    print(f"  {'-'*42}")
    accuracies = []
    for i in range(len(targets)):
        pct_err = abs(preds[i] - targets[i]) / targets[i] * 100
        accuracy = 100.0 - pct_err
        accuracies.append(accuracy)
        print(f"  {i+1:>6}  {preds[i]:>10.4f}  {targets[i]:>10.4f}  {accuracy:>9.2f}%")
    mean_accuracy = np.mean(accuracies)
    print(f"  {'-'*42}")
    print(f"  Mean accuracy: {mean_accuracy:.2f}%")
    print(f"  RMSE: {final_rmse:.4f} mm  |  R²: {final_r2:.4f}")
    print(f"{'='*60}")

    # ── loss curve plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label="Train MSE")
    axes[0].plot(test_losses, label="Test MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Training & Test Loss")
    axes[0].legend()

    # ── predicted vs actual scatter ──
    axes[1].scatter(targets, preds, alpha=0.7, edgecolors="k", linewidths=0.5)
    lo = min(targets.min(), preds.min()) * 0.9
    hi = max(targets.max(), preds.max()) * 1.1
    axes[1].plot([lo, hi], [lo, hi], "r--", label="Ideal (y=x)")
    axes[1].set_xlabel("Actual Vbmax (mm)")
    axes[1].set_ylabel("Predicted Vbmax (mm)")
    axes[1].set_title(f"Predicted vs Actual  (R²={final_r2:.3f})")
    axes[1].legend()
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "results.png", dpi=150)
    print(f"\nPlot saved to results.png")

    # ── save model ──
    torch.save(model.state_dict(), PROJECT_ROOT / "model.pth")
    print(f"Model saved to model.pth")


if __name__ == "__main__":
    main()
