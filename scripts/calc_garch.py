import numpy as np
import pandas as pd
import os
from arch import arch_model
from scipy.stats import t as student_t
from tqdm import tqdm

# Define directories
INPUT_DIR = "asset_data"
OUTPUT_DIR = "garch_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define assets and quantiles
ASSETS = ['s&p500', 'dowjones', 'ftse100', 'oilspot', 'goldspot', 'usdjpy']
QUANTILES = [0.01, 0.025, 0.05]
ROLLING_WINDOW = 512  # 512-day rolling window for GARCH estimation

def compute_garch_var_es(returns, window_size=ROLLING_WINDOW, alphas=QUANTILES):
    """
    Compute rolling GARCH(1,1) model with skewed-t distribution and extract VaR & ES.
    """
    results = {alpha: {'VaR': [], 'ES': []} for alpha in alphas}

    for i in tqdm(range(window_size, len(returns)), desc="Computing GARCH-based VaR & ES"):
        try:
            train_data = returns.iloc[i-window_size:i]
            model = arch_model(train_data, vol='Garch', p=1, q=1, dist='skewt')
            res = model.fit(disp='off', show_warning=False)

            # Get one-step ahead forecast
            forecast = res.forecast(horizon=1, reindex=False)
            garch_vol = np.sqrt(forecast.variance.values[-1, 0])

            # Check if 'nu' exists in model parameters
            if 'nu' in res.params and res.params['nu'] > 2:
                nu_value = res.params['nu']
                std_dev = np.sqrt(nu_value / (nu_value - 2))
                for alpha in alphas:
                    student_t_quantile = student_t.ppf(alpha, nu_value)
                    var_t = garch_vol * student_t_quantile / std_dev

                    # Compute Expected Shortfall (ES)
                    es_t = -garch_vol * (student_t.pdf(student_t_quantile, nu_value) / alpha) * \
                          ((nu_value + student_t_quantile**2) / (nu_value - 1)) / std_dev

                    results[alpha]['VaR'].append(var_t)
                    results[alpha]['ES'].append(es_t)
            else:
                print(f"Warning: 'nu' missing or invalid at index {i}, falling back to normal distribution")
                fallback_df = 100  # Large degrees of freedom to approximate normal behavior
                for alpha in alphas:
                    var_t = garch_vol * student_t.ppf(alpha, fallback_df)
                    es_t = -garch_vol * (student_t.pdf(student_t.ppf(alpha, fallback_df), fallback_df) / alpha)
                    results[alpha]['VaR'].append(var_t)
                    results[alpha]['ES'].append(es_t)

        except Exception as e:
            print(f"Exception at index {i}: {str(e)}")
            for alpha in alphas:
                results[alpha]['VaR'].append(np.nan)
                results[alpha]['ES'].append(np.nan)

    # Pad the beginning with NaN to align timestamps correctly
    for alpha in alphas:
        results[alpha]['VaR'] = [np.nan] * window_size + results[alpha]['VaR']
        results[alpha]['ES'] = [np.nan] * window_size + results[alpha]['ES']

    return results



def process_asset(asset_name):
    """
    Process each asset: Load dataset, compute GARCH(1,1) VaR & ES, and save results.
    """
    input_file = os.path.join(INPUT_DIR, f"{asset_name}_data.csv")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Load data
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    returns = df[f"{asset_name}_log_returns"].dropna()
    
    print(f"Processing {asset_name}: {len(returns)} records")
    
    # Compute GARCH-based VaR & ES
    risk_metrics = compute_garch_var_es(returns)
    
    # Save results
    df_var_es = pd.DataFrame(index=returns.index)
    for alpha in QUANTILES:
        df_var_es[f"VaR_{int(alpha*100)}"] = risk_metrics[alpha]['VaR']
        df_var_es[f"ES_{int(alpha*100)}"] = risk_metrics[alpha]['ES']
    
    output_file = os.path.join(OUTPUT_DIR, f"{asset_name}_garch_var_es.csv")
    df_var_es.to_csv(output_file)
    print(f"✓ Saved GARCH VaR & ES for {asset_name} to {output_file}")

def main():
    for asset in ASSETS[1:]:
        process_asset(asset)

if __name__ == "__main__":
    main()
