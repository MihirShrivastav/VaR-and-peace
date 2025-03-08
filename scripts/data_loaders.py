import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler  # Using RobustScaler for targets

# === CONFIGURATION ===
ASSETS = ["s&p500", "goldspot", "oilspot", "dowjones", "ftse100", "usdjpy"]
ROLLING_WINDOW = 512  # First 512 entries don't have VaR/ES estimates
SRNN_WINDOW = 32      # SRNN models use 32-day window for predictions
BATCH_SIZE = 32
QUANTILE_LEVEL = "1"  # Options: "1" (1%), "2" (2.5%), "5" (5%)

SCALER_DIR = "asset_data/scalers/"
os.makedirs(SCALER_DIR, exist_ok=True)

def visualize_scaling(raw_values, scaled_values, target_name, asset):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(raw_values, bins=30, alpha=0.7)
    plt.title(f"{asset} Raw {target_name}")
    plt.subplot(1,2,2)
    plt.hist(scaled_values, bins=30, alpha=0.7, color='orange')
    plt.title(f"{asset} Scaled {target_name}")
    plt.tight_layout()
    plt.show()

def load_asset_data(asset, quantile_level=QUANTILE_LEVEL, visualize=False):
    print(f"\n📌 Loading data for {asset} at {quantile_level}% quantile...")

    # Load preprocessed squared returns (already scaled)
    df = pd.read_csv(f"asset_data/{asset}_data.csv", index_col=0)
    returns = df[[f"{asset}_squared_returns"]].values  # Already scaled squared returns

    # Load true VaR & ES at selected quantile level
    var_es_df = pd.read_csv(f"garch_results/{asset}_garch_var_es.csv", index_col=0)
    var_col = f"VaR_{quantile_level}"
    es_col = f"ES_{quantile_level}"

    if var_col not in var_es_df.columns or es_col not in var_es_df.columns:
        raise ValueError(f"❌ Column names {var_col} and {es_col} not found in {asset} GARCH file!")

    var_es = var_es_df[[var_col, es_col]].values  # True VaR & ES

    # Drop first ROLLING_WINDOW entries to align dataset
    returns = returns[ROLLING_WINDOW:]
    var_es = var_es[ROLLING_WINDOW:]

    assert len(returns) == len(var_es), f"❌ Data misalignment for {asset}: returns({len(returns)}) != var_es({len(var_es)})"

    # Create sequences for SRNN models
    X_sequences, y_targets = [], []
    for i in range(len(returns) - SRNN_WINDOW):
        X_sequences.append(returns[i:i + SRNN_WINDOW])  # Input sequence (already scaled)
        y_targets.append(var_es[i + SRNN_WINDOW])         # Target: Next day VaR & ES

    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    # Split into train-test sets (80% train, 20% test)
    split_idx = int(0.8 * len(X_sequences))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

    # Print raw values before scaling for targets
    print("\n=== BEFORE SCALING ===")
    print(f"VaR first 5 values: {y_train[:5, 0]}")
    print(f"ES first 5 values: {y_train[:5, 1]}")

    # Do NOT scale input features (squared returns) because they are already scaled
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Scale the target values (VaR and ES) using RobustScaler
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Save scaling parameters for targets (using median and IQR)
    np.savez(f"{SCALER_DIR}/{asset}_scaler.npz",
             X_info=None,
             y_median=y_scaler.center_,
             y_iqr=y_scaler.scale_)

    # Visualizations for targets if requested
    if visualize:
        visualize_scaling(y_train[:, 0], y_train_scaled[:, 0], "VaR", asset)
        visualize_scaling(y_train[:, 1], y_train_scaled[:, 1], "ES", asset)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n=== AFTER SCALING ===")
    print(f"VaR first 5 values (scaled): {y_train_scaled[:5, 0]}")
    print(f"ES first 5 values (scaled): {y_train_scaled[:5, 1]}")

    print(f"\n✅ {asset.upper()} - Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "X_train": X_train_tensor,
        "y_train": y_train_tensor,
        "X_test": X_test_tensor,
        "y_test": y_test_tensor,
        "X_info": None,
        "y_median": [float(val) for val in y_scaler.center_],
        "y_iqr": [float(val) for val in y_scaler.scale_],
        "sequence_length": SRNN_WINDOW
    }

def save_data_loaders(asset_data, output_path="asset_data/data_loaders.pth"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable_data = {}
    for asset, data in asset_data.items():
        serializable_data[asset] = {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "batch_size": BATCH_SIZE,
            "X_info": data["X_info"],
            "y_median": data["y_median"],
            "y_iqr": data["y_iqr"],
            "sequence_length": data["sequence_length"]
        }
    torch.save(serializable_data, output_path, pickle_protocol=2)
    print(f"\n✅ Asset data saved to {output_path}")

if __name__ == "__main__":
    # For demonstration, we enable visualization for the first asset
    asset_data = {asset: load_asset_data(asset, visualize=(asset == ASSETS[0])) for asset in ASSETS}
    save_data_loaders(asset_data)
