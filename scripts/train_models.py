import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# === LOAD PREPROCESSED DATA ===
print("\n📌 Loading preprocessed data...")
try:
    asset_data = torch.load("asset_data/data_loaders.pth", weights_only=False)
    print("✅ Successfully loaded data with weights_only=False")
except Exception as e:
    print(f"Error loading data: {e}")
    import torch.serialization
    torch.serialization.add_safe_globals(['numpy.core.multiarray', 'numpy.core.multiarray.scalar'])
    asset_data = torch.load("asset_data/data_loaders.pth", weights_only=False)
    print("✅ Successfully loaded data with safe globals")

# Create DataLoaders and Scaling Info
asset_loaders, scaling_info, sequence_lengths = {}, {}, {}
for asset, data in asset_data.items():
    train_loader = DataLoader(TensorDataset(data["X_train"], data["y_train"]), batch_size=data["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(data["X_test"], data["y_test"]), batch_size=data["batch_size"], shuffle=False)

    asset_loaders[asset] = (train_loader, test_loader)
    scaling_info[asset] = {"X_info": data["X_info"], "y_median": data["y_median"], "y_iqr": data["y_iqr"]}
    sequence_lengths[asset] = data["sequence_length"]

print(f"✅ Created DataLoaders for {len(asset_loaders)} assets")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODELS_DIR = "models/"
os.makedirs(MODELS_DIR, exist_ok=True)

# === FZ0 LOSS FUNCTION (Improved) ===
class FZ0Loss(nn.Module):
    def __init__(self, alpha=0.01, epsilon=1e-6):
        super(FZ0Loss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon  

    def forward(self, y_true, vaR_pred, es_pred):
        es_pred = torch.clamp(es_pred, min=-10, max=-self.epsilon)
        vaR_pred = torch.clamp(vaR_pred, min=-10, max=10)

        indicator = (y_true <= vaR_pred).float()
        
        term1 = -1.0 / (es_pred - self.epsilon)
        term2 = (1.0 / self.alpha) * (y_true * indicator) - es_pred
        term3 = vaR_pred / (es_pred - self.epsilon)
        term4 = torch.log(-(es_pred - self.epsilon))

        loss = term1 * term2 + term3 + term4 - 1
        loss = loss / torch.abs(loss).mean()

        # Additional loss penalty for large prediction deviations
        deviation_penalty = 0.5 * torch.mean((vaR_pred - y_true) ** 2)
        confidence_penalty = 0.1 * torch.mean(torch.abs(vaR_pred - y_true))
        return loss.mean() + deviation_penalty + confidence_penalty

# === MODEL DEFINITIONS ===
class SRNN_VE(nn.Module):
    def __init__(self, variant="VE1", input_size=1, hidden_size=10, output_size=2, dropout=0.2):
        super(SRNN_VE, self).__init__()
        self.hidden_size = hidden_size
        self.variant = variant
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, h=None):
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        h, _ = self.rnn(x, h)

        if self.variant == "VE2":
            h_transformed = torch.sqrt(torch.abs(h[:, -1, :]) + 1e-8)
            out = self.fc(h_transformed)
        elif self.variant == "VE3":
            h_transformed = torch.sqrt(torch.abs(h[:, -1, :]) + 1e-8)
            v1 = self.fc(h[:, -1, :])
            v2 = self.fc(h_transformed)
            out = -torch.abs(v1 + v2)
        else:
            out = self.fc(h[:, -1, :])

        return out, h

# === TRAINING FUNCTION (Training VE3 Only) ===
def train_model(asset, variant="VE3", num_epochs=50, learning_rate=0.0001):
    print(f"\n[Training {variant} for {asset}]")

    train_loader, test_loader = asset_loaders[asset]
    scale_info = scaling_info[asset]

    model = SRNN_VE(variant=variant).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = FZ0Loss(alpha=0.01)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out, _ = model(X_batch)

            vaR_pred, es_pred = out[:, 0], out[:, 1]
            true_var = y_batch[:, 0]

            if epoch % 10 == 0 and valid_batches == 0:
                print(f"\nEpoch {epoch + 1} Sample Predictions:")
                print(f"  Raw Predicted VaR: {vaR_pred[:5].detach().cpu().numpy()}")
                print(f"  Raw Predicted ES: {es_pred[:5].detach().cpu().numpy()}")
                print(f"  True VaR: {true_var[:5].detach().cpu().numpy()}")

            loss = loss_fn(true_var, vaR_pred, es_pred)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue  

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

        avg_epoch_loss = epoch_loss / max(1, valid_batches)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")

    model_path = f"{MODELS_DIR}/{variant}_{asset}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved: {model_path}")


# === EVALUATION FUNCTION ===
def evaluate_model(asset, variant="VE3"):
    print(f"\n[Evaluating {variant} for {asset}]")
    
    # Create results directory if it doesn't exist
    RESULTS_DIR = f"results/{asset}/{variant}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model_path = f"{MODELS_DIR}/{variant}_{asset}.pth"
    model = SRNN_VE(variant=variant).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    _, test_loader = asset_loaders[asset]
    scale_info = scaling_info[asset]
    y_median, y_iqr = scale_info["y_median"], scale_info["y_iqr"]
    
    # Lists to store all predictions and actual values
    all_var_pred = []
    all_es_pred = []
    all_var_true = []
    all_es_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out, _ = model(X_batch)
            
            # Extract predictions
            var_pred_scaled = out[:, 0].cpu().numpy()
            es_pred_scaled = out[:, 1].cpu().numpy()
            
            # Unscale predictions using RobustScaler parameters
            var_pred_unscaled = (var_pred_scaled * y_iqr[0]) + y_median[0]
            es_pred_unscaled = (es_pred_scaled * y_iqr[1]) + y_median[1]
            true_var_unscaled = (y_batch[:, 0].cpu().numpy() * y_iqr[0]) + y_median[0]
            true_es_unscaled = (y_batch[:, 1].cpu().numpy() * y_iqr[1]) + y_median[1]
            
            # Append to lists
            all_var_pred.extend(var_pred_unscaled)
            all_es_pred.extend(es_pred_unscaled)
            all_var_true.extend(true_var_unscaled)
            all_es_true.extend(true_es_unscaled)
    
    # Convert to numpy arrays for easier manipulation
    all_var_pred = np.array(all_var_pred)
    all_es_pred = np.array(all_es_pred)
    all_var_true = np.array(all_var_true)
    all_es_true = np.array(all_es_true)
    
    # Calculate metrics
    var_mse = mean_squared_error(all_var_true, all_var_pred)
    var_mae = mean_absolute_error(all_var_true, all_var_pred)
    var_r2 = r2_score(all_var_true, all_var_pred)
    
    es_mse = mean_squared_error(all_es_true, all_es_pred)
    es_mae = mean_absolute_error(all_es_true, all_es_pred)
    es_r2 = r2_score(all_es_true, all_es_pred)
    
    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"VaR MSE: {var_mse:.6f}")
    print(f"VaR MAE: {var_mae:.6f}")
    print(f"VaR R²: {var_r2:.6f}")
    print(f"ES MSE: {es_mse:.6f}")
    print(f"ES MAE: {es_mae:.6f}")
    print(f"ES R²: {es_r2:.6f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'R²'],
        'VaR': [var_mse, var_mae, var_r2],
        'ES': [es_mse, es_mae, es_r2]
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print(f"✅ Metrics saved to {RESULTS_DIR}/metrics.csv")
    
    # Create visualizations
    
    # 1. Scatter plot of predicted vs actual values
    plt.figure(figsize=(12, 5))
    
    # VaR scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(all_var_true, all_var_pred, alpha=0.5)
    plt.plot([min(all_var_true), max(all_var_true)], [min(all_var_true), max(all_var_true)], 'r--')
    plt.xlabel('True VaR')
    plt.ylabel('Predicted VaR')
    plt.title(f'VaR Prediction (R² = {var_r2:.4f})')
    
    # ES scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(all_es_true, all_es_pred, alpha=0.5)
    plt.plot([min(all_es_true), max(all_es_true)], [min(all_es_true), max(all_es_true)], 'r--')
    plt.xlabel('True ES')
    plt.ylabel('Predicted ES')
    plt.title(f'ES Prediction (R² = {es_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/scatter_plot.png", dpi=300)
    print(f"✅ Scatter plot saved to {RESULTS_DIR}/scatter_plot.png")
    
    # 2. Time series plot of the first 100 predictions
    n_samples = min(100, len(all_var_true))
    x_range = np.arange(n_samples)
    
    plt.figure(figsize=(12, 8))
    
    # VaR time series
    plt.subplot(2, 1, 1)
    plt.plot(x_range, all_var_true[:n_samples], label='True VaR', color='blue')
    plt.plot(x_range, all_var_pred[:n_samples], label='Predicted VaR', color='red', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('VaR')
    plt.title('VaR: True vs Predicted (First 100 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ES time series
    plt.subplot(2, 1, 2)
    plt.plot(x_range, all_es_true[:n_samples], label='True ES', color='blue')
    plt.plot(x_range, all_es_pred[:n_samples], label='Predicted ES', color='red', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('ES')
    plt.title('ES: True vs Predicted (First 100 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/time_series.png", dpi=300)
    print(f"✅ Time series plot saved to {RESULTS_DIR}/time_series.png")
    
    # 3. Error distribution histogram
    var_errors = all_var_pred - all_var_true
    es_errors = all_es_pred - all_es_true
    
    plt.figure(figsize=(12, 5))
    
    # VaR error histogram
    plt.subplot(1, 2, 1)
    plt.hist(var_errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('VaR Prediction Error Distribution')
    
    # ES error histogram
    plt.subplot(1, 2, 2)
    plt.hist(es_errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('ES Prediction Error Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/error_distribution.png", dpi=300)
    print(f"✅ Error distribution plot saved to {RESULTS_DIR}/error_distribution.png")
    
    # 4. Save predictions to CSV for further analysis
    results_df = pd.DataFrame({
        'True_VaR': all_var_true,
        'Predicted_VaR': all_var_pred,
        'VaR_Error': var_errors,
        'True_ES': all_es_true,
        'Predicted_ES': all_es_pred,
        'ES_Error': es_errors
    })
    results_df.to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)
    print(f"✅ Prediction results saved to {RESULTS_DIR}/predictions.csv")
    
    return {
        'var_metrics': {'mse': var_mse, 'mae': var_mae, 'r2': var_r2},
        'es_metrics': {'mse': es_mse, 'mae': es_mae, 'r2': es_r2}
    }


# === MAIN SCRIPT ===
if __name__ == "__main__":
    for asset in asset_loaders.keys():
        for variant in ["VE3"]:
            train_model(asset, variant)
            evaluate_model(asset, variant)
