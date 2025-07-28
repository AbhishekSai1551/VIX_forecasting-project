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
    """Download VIX and VVIX data from Yahoo Finance"""
    tickers = ['^VIX', '^VVIX']
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    vix_data = data['^VIX'].copy()
    vvix_data = data['^VVIX'].copy()
    
    # Align data by common dates
    common_dates = vix_data.index.intersection(vvix_data.index)
    vix_data = vix_data.loc[common_dates]
    vvix_data = vvix_data.loc[common_dates]
    
    return vix_data, vvix_data

def robust_data_cleaning(df):
    """Enhanced cleaning with mean imputation, 3-sigma outlier handling, and inf/NaN checks"""
    clean_df = df.copy()
    
    # Replace inf with NaN
    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Mean imputation for missing values
    for col in clean_df.columns:
        if clean_df[col].isnull().sum() > 0:
            col_mean = clean_df[col].mean()
            clean_df[col] = clean_df[col].fillna(col_mean)
            print(f"Imputed {clean_df[col].isnull().sum()} missing values in {col} with mean {col_mean:.2f}")
    
    # 3-sigma outlier detection and treatment
    outlier_counts = {}
    for col in clean_df.select_dtypes(include=[np.number]).columns:
        z_scores = zscore(clean_df[col].replace([np.inf, -np.inf], np.nan).dropna())
        outliers = np.abs(z_scores) > 3
        if outliers.any():
            upper_bound = clean_df[col].mean() + 3*clean_df[col].std()
            lower_bound = clean_df[col].mean() - 3*clean_df[col].std()
            clean_df.loc[outliers, col] = np.clip(clean_df.loc[outliers, col], lower_bound, upper_bound)
            outlier_counts[col] = outliers.sum()
    
    if outlier_counts:
        print("\nOutliers treated:")
        for col, count in outlier_counts.items():
            print(f"{col}: {count} outliers winsorized")
    
    # Final check for inf/NaN
    if clean_df.isin([np.inf, -np.inf]).any().any() or clean_df.isnull().any().any():
        raise ValueError("Data still contains inf/NaN after cleaning")
    
    return clean_df

def create_technical_features(df):
    """Create technical features with safeguards against inf/NaN"""
    enhanced_df = df.copy()
    
    # Moving Averages
    for window in [10, 50, 100]:
        enhanced_df[f'MA_{window}_VIX'] = enhanced_df['Close_VIX'].rolling(window).mean()
        enhanced_df[f'MA_{window}_VVIX'] = enhanced_df['Close_VVIX'].rolling(window).mean()
    
    # Yield Calculations with clipping to avoid extreme values
    enhanced_df['Yield_VIX'] = enhanced_df['Close_VIX'].pct_change() * 100
    enhanced_df['Yield_VIX'] = enhanced_df['Yield_VIX'].clip(-100, 100)
    enhanced_df['Yield_VVIX'] = enhanced_df['Close_VVIX'].pct_change() * 100
    enhanced_df['Yield_VVIX'] = enhanced_df['Yield_VVIX'].clip(-100, 100)
    
    # Volatility Measures
    enhanced_df['Volatility_VIX'] = enhanced_df['Yield_VIX'].rolling(5).std()
    enhanced_df['Volatility_VVIX'] = enhanced_df['Yield_VVIX'].rolling(5).std()
    
    # Lagged Features
    for lag in [1, 3, 5]:
        enhanced_df[f'Lag_{lag}_VIX'] = enhanced_df['Close_VIX'].shift(lag)
        enhanced_df[f'Lag_{lag}_VVIX'] = enhanced_df['Close_VVIX'].shift(lag)
    
    # GARCH Conditional Volatility with safeguards
    try:
        returns = np.log((enhanced_df['Close_VIX'] + 1e-6) / (enhanced_df['Close_VIX'].shift(1) + 1e-6)).replace([np.inf, -np.inf], np.nan).dropna()
        garch = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
        garch_fit = garch.fit(disp='off')
        enhanced_df['GARCH_Volatility'] = np.nan
        enhanced_df.loc[garch_fit.conditional_volatility.index, 'GARCH_Volatility'] = garch_fit.conditional_volatility
        enhanced_df['GARCH_Volatility'] = enhanced_df['GARCH_Volatility'].clip(0, 100)  # Cap volatility
    except:
        enhanced_df['GARCH_Volatility'] = enhanced_df['Volatility_VIX']
    
    # Drop NAs and check for inf/NaN
    enhanced_df.dropna(inplace=True)
    if enhanced_df.isin([np.inf, -np.inf]).any().any():
        raise ValueError("Feature engineering produced inf values")
    
    return enhanced_df

def prepare_sequences(df, sequence_length=30, target_col='Close_VIX'):
    """Prepare sequences for deep learning models"""
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    features = df[feature_cols].values
    target = df[target_col].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    return X_scaled, y, feature_cols, scaler

def create_sequences(data, n_steps=30):
    """Create sequences for time series modeling (legacy function)"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i-n_steps:i, :-1].values)
        y.append(data.iloc[i, -1])
    return np.array(X), np.array(y)

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

def fit_duan_garch_model(returns_data):
    """Implement Duan's GARCH methodology for VIX forecasting"""
    try:
        # GARCH(1,1) with Student-t distribution
        model = arch_model(returns_data, vol='Garch', p=1, q=1, dist='t', rescale=False)
        fitted_model = model.fit(disp='off')
        return fitted_model
    except Exception as e:
        print(f"GARCH model fitting failed: {e}")
        return None

def garch_forecast_vix(fitted_model, horizon=1):
    """Generate VIX forecasts using fitted GARCH model"""
    if fitted_model is None:
        return None
    try:
        forecast = fitted_model.forecast(horizon=horizon)
        return forecast
    except Exception as e:
        print(f"GARCH forecasting failed: {e}")
        return None

def create_features(vix_data, vvix_data):
    """Create combined features from VIX and VVIX data"""
    # Merge VIX and VVIX data
    raw_data = pd.merge(vix_data, vvix_data, left_index=True, right_index=True,
                        suffixes=('_VIX', '_VVIX'))
    
    # Clean the data
    cleaned_data = robust_data_cleaning(raw_data)
    
    # Create technical features
    featured_data = create_technical_features(cleaned_data)
    
    return featured_data

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """Comprehensive model performance evaluation"""
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R²: {metrics['R2']:.6f}")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.3f}")
    
    return metrics

def save_model_results(results, filename):
    """Save model results to pickle file"""
    import joblib
    joblib.dump(results, filename)
    print(f"Results saved to {filename}")

def load_model_results(filename):
    """Load model results from pickle file"""
    import joblib
    try:
        results = joblib.load(filename)
        print(f"Results loaded from {filename}")
        return results
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

print("VIX Research Utility Functions loaded successfully!")
