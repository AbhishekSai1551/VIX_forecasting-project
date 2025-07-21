# Shared functions for data processing, evaluation metrics, and statistical tests

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def download_market_data():
    """Download VIX and VVIX data"""
    tickers = ['^VIX', '^VVIX']
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    vix_data = data['^VIX'].copy()
    vvix_data = data['^VVIX'].copy()
    
    common_dates = vix_data.index.intersection(vvix_data.index)
    vix_data = vix_data.loc[common_dates]
    vvix_data = vvix_data.loc[common_dates]
    
    return vix_data, vvix_data

def clean_data(df):
    """data cleaning"""
    clean_df = df.copy()
    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in clean_df.columns:
        if clean_df[col].isnull().sum() > 0:
            col_mean = clean_df[col].mean()
            clean_df[col] = clean_df[col].fillna(col_mean)
    
    for col in clean_df.select_dtypes(include=[np.number]).columns:
        z_scores = zscore(clean_df[col].replace([np.inf, -np.inf], np.nan).dropna())
        outliers = np.abs(z_scores) > 3
        if outliers.any():
            upper_bound = clean_df[col].mean() + 3*clean_df[col].std()
            lower_bound = clean_df[col].mean() - 3*clean_df[col].std()
            clean_df.loc[outliers, col] = np.clip(clean_df.loc[outliers, col], lower_bound, upper_bound)
    
    return clean_df

def create_technical_features(df):
    """Create technical features"""
    enhanced_df = df.copy()
    
    # Moving Averages
    for window in [10, 50, 100]:
        enhanced_df[f'MA_{window}_VIX'] = enhanced_df['Close_VIX'].rolling(window).mean()
        enhanced_df[f'MA_{window}_VVIX'] = enhanced_df['Close_VVIX'].rolling(window).mean()
    
    # Returns and volatility
    enhanced_df['Yield_VIX'] = enhanced_df['Close_VIX'].pct_change() * 100
    enhanced_df['Yield_VIX'] = enhanced_df['Yield_VIX'].clip(-100, 100)
    enhanced_df['Yield_VVIX'] = enhanced_df['Close_VVIX'].pct_change() * 100
    enhanced_df['Yield_VVIX'] = enhanced_df['Yield_VVIX'].clip(-100, 100)
    
    enhanced_df['Volatility_VIX'] = enhanced_df['Yield_VIX'].rolling(5).std()
    enhanced_df['Volatility_VVIX'] = enhanced_df['Yield_VVIX'].rolling(5).std()
    
    # Lagged features
    for lag in [1, 3, 5]:
        enhanced_df[f'Lag_{lag}_VIX'] = enhanced_df['Close_VIX'].shift(lag)
        enhanced_df[f'Lag_{lag}_VVIX'] = enhanced_df['Close_VVIX'].shift(lag)
    
    # GARCH volatility
    try:
        returns = np.log((enhanced_df['Close_VIX'] + 1e-6) / (enhanced_df['Close_VIX'].shift(1) + 1e-6)).replace([np.inf, -np.inf], np.nan).dropna()
        garch = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
        garch_fit = garch.fit(disp='off')
        enhanced_df['GARCH_Volatility'] = np.nan
        enhanced_df.loc[garch_fit.conditional_volatility.index, 'GARCH_Volatility'] = garch_fit.conditional_volatility
        enhanced_df['GARCH_Volatility'] = enhanced_df['GARCH_Volatility'].clip(0, 100)
    except:
        enhanced_df['GARCH_Volatility'] = enhanced_df['Volatility_VIX']
    
    enhanced_df.dropna(inplace=True)
    return enhanced_df

def optimize_features(df, target_col='Close_VIX', variance_threshold=0.95):
    """Feature selection with PCA"""
    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    selected_features = target_corr[target_corr > 0.5].index.tolist()
    
    X = df[selected_features]
    y = df[target_col]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_cols = [f'PC_{i+1}' for i in range(pca.n_components_)]
    final_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    final_df[target_col] = y
    
    return final_df, pca, scaler, selected_features

def create_sequences(data, n_steps=30):
    """Create sequences for time series modeling"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i-n_steps:i, :-1].values)
        y.append(data.iloc[i, -1])
    return np.array(X), np.array(y)

def time_series_split(X, y, n_splits=5):
    """Create time series cross-validation splits"""
    splits = []
    n_samples = len(X)
    test_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = test_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + test_size // 2, n_samples)
        
        if val_end > val_start:
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))
    
    return splits

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

def diebold_mariano_test(errors1, errors2, h=1):
    """Diebold-Mariano test for forecast accuracy comparison"""
    d = errors1**2 - errors2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    if d_var == 0:
        return 0, 1.0
    
    # Newey-West adjustment for autocorrelation
    n = len(d)
    gamma_0 = d_var
    gamma_sum = 0
    
    for j in range(1, h):
        if j < n:
            gamma_j = np.cov(d[:-j], d[j:])[0, 1]
            gamma_sum += 2 * gamma_j
    
    long_run_var = gamma_0 + gamma_sum
    dm_stat = d_mean / np.sqrt(long_run_var / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def calculate_confidence_intervals(predictions, confidence_level=0.95):
    """Calculate prediction confidence intervals using bootstrap"""
    n_bootstrap = 1000
    bootstrap_preds = []
    
    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(predictions), size=len(predictions), replace=True)
        bootstrap_preds.append(predictions[bootstrap_idx])
    
    bootstrap_preds = np.array(bootstrap_preds)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_preds, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_preds, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

def create_baseline_models(y_train, y_test):
    """Create baseline forecast models for comparison"""
    baselines = {}
    
    # Naive forecast (persistence)
    naive_pred = np.full(len(y_test), y_train[-1])
    baselines['Naive'] = naive_pred
    
    # Moving average
    window = min(30, len(y_train))
    ma_pred = np.full(len(y_test), np.mean(y_train[-window:]))
    baselines['Moving_Average'] = ma_pred
    
    # Random walk
    rw_pred = np.full(len(y_test), y_train[-1])
    baselines['Random_Walk'] = rw_pred
    
    return baselines

print("VIX Research Utility Functions loaded successfully!")
