# VIX Forecasting Research Project

## Research Objective

This project conducts a comprehensive research study to **study the capabilities of modern machine learning techniques for predicting next-day VIX (Volatility Index) values**. The research compares advanced deep learning architectures (CNN-LSTM and GRU) against traditional econometric models (GARCH) to evaluate their forecasting performance.

## Research Methodology

### 1. Model Architecture Optimization
- **CNN-LSTM Hyperparameter Optimization**: Systematic optimization using Optuna framework
- **GRU Hyperparameter Optimization**: Comprehensive parameter tuning for optimal performance
- **Architecture Comparison**: Evaluation of different neural network configurations

### 2. Baseline Comparison
- **GARCH Model**: Implementation of Duan's GARCH methodology for volatility forecasting
- **Performance Benchmarking**: Statistical comparison of modern ML vs traditional econometric approaches

### 3. Evaluation Period
- **Training Period**: Historical VIX data through May 2025
- **Evaluation Period**: June 1, 2025 to present
- **Prediction Horizon**: Next-day VIX forecasting

## Project Structure

```
VIX_forecasting_project/
├── Enhanced_CNN_LSTM_Optimization.ipynb          # Enhanced CNN-LSTM optimization with statistical rigor
├── Enhanced_GRU_Optimization.ipynb               # Enhanced GRU optimization with statistical rigor
├── Enhanced_VIX_Comprehensive_Analysis.ipynb     # Complete enhanced analysis with significance testing
├── vix_research_utils.py                         # Shared utility functions for all notebooks
├── requirements.txt                               # Python dependencies
├── README.md                                      # This documentation
├── Enhanced_VIX_Model_Results_and_Errors_from_June2025.csv # Enhanced research results (generated)
└── venv/                                          # Virtual environment
```

## Setup Instructions

### Prerequisites
- **Python 3.8+** (recommended: Python 3.8-3.10)
- **8GB+ RAM** (for model training)
- **GPU recommended** (for faster training, optional)

### Installation

1. **Create and activate virtual environment**:
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

## Research Execution

### Research Methodology
1. **CNN-LSTM optimization**: `Enhanced_CNN_LSTM_Optimization.ipynb`
   - Time series cross-validation with 5 folds
   - 200 trials per architecture for thorough search
   - Statistical significance testing vs baselines
   - Confidence interval estimation

2. **Enhanced GRU optimization**: `Enhanced_GRU_Optimization.ipynb`
   - Expanded hyperparameter search spaces
   - Bayesian optimization with TPE sampler
   - Enhanced regularization techniques
   - Comprehensive evaluation metrics

3. **Enhanced comprehensive analysis**: `Enhanced_VIX_Comprehensive_Analysis.ipynb`
   - Diebold-Mariano statistical significance tests
   - Baseline model comparisons (naive, moving average, random walk)
   - Confidence intervals for predictions
   - Enhanced visualization and results export

## Enhanced Research Notebooks

### 1. Enhanced CNN-LSTM Hyperparameter Optimization
**File**: `Enhanced_CNN_LSTM_Optimization.ipynb`

**Purpose**: Advanced CNN-LSTM optimization with statistical rigor
- **Methodology**: Time series cross-validation, 200 trials per architecture
- **Search spaces**: Learning rates (1e-5 to 1e-1), weight decay, attention parameters
- **Architecture variants**: Enhanced Basic CNN-LSTM, Enhanced Attention CNN-LSTM
- **Statistical analysis**: Diebold-Mariano tests, confidence intervals, baseline comparisons
- **Output**: Statistically validated optimal CNN-LSTM configurations

### 2. Enhanced GRU Hyperparameter Optimization
**File**: `Enhanced_GRU_Optimization.ipynb`

**Purpose**: Advanced GRU optimization with enhanced methodology
- **Enhanced features**: Bidirectional processing, self-attention mechanisms, residual connections
- **Expanded hyperparameters**: Recurrent dropout, attention dropout, weight decay optimization
- **Architecture variants**: Enhanced Basic GRU, Enhanced Bidirectional GRU, Enhanced Attention GRU
- **Statistical validation**: Cross-validation, significance testing, performance benchmarking
- **Output**: Statistically validated optimal GRU configurations

### 3. Enhanced VIX Comprehensive Forecasting Analysis
**File**: `Enhanced_VIX_Comprehensive_Analysis.ipynb`

**Purpose**: Complete enhanced research analysis with statistical rigor
- **Enhanced GARCH**: Better error handling, simulation-based forecasting
- **Advanced models**: Bidirectional LSTM/GRU with multi-head attention, Huber loss
- **Statistical testing**: Diebold-Mariano tests for forecast accuracy comparison
- **Baseline comparisons**: Naive, moving average, random walk models
- **Confidence intervals**: Bootstrap-based prediction intervals
- **Enhanced visualization**: Comprehensive results plots and statistical summaries
- **Output**: Statistically validated research results with significance testing

### 4. Shared Research Utilities
**File**: `vix_research_utils.py`

**Purpose**: Common functions for data processing, evaluation, and statistical testing
- **Data functions**: Market data download, cleaning, feature engineering, PCA optimization
- **Evaluation metrics**: MSE, MAE, RMSE, R², directional accuracy
- **Statistical tests**: Diebold-Mariano test, confidence intervals, baseline models
- **Time series utilities**: Cross-validation splits, sequence creation
- **Output**: Reusable functions ensuring consistency across all notebooks

## Expected Research Outcomes

### Enhanced Model Performance Metrics
- **Enhanced CNN-LSTM**: R² = 0.75-0.85, statistically significant improvement over baselines
- **Enhanced GRU**: R² = 0.70-0.80, statistically significant improvement over baselines
- **GARCH**: Traditional econometric baseline with enhanced forecasting methodology
- **Directional Accuracy**: 70-80% for next-day VIX movement prediction
- **Statistical Significance**: Diebold-Mariano test p-values < 0.05 vs baselines

### Enhanced Training Time Estimates
- **Enhanced hyperparameter optimization**: 3-6 hours per model (200 trials with CV)
- **Enhanced model training**: 20-40 minutes per model with early stopping
- **Complete enhanced analysis**: 6-10 hours total runtime
- **Statistical testing**: Additional 30-60 minutes for significance tests

### Enhanced Research Deliverables
- **Statistically validated hyperparameters** with confidence intervals
- **Rigorous performance comparison** with significance testing
- **Enhanced prediction results** with confidence intervals for June 2025-present
- **Comprehensive statistical analysis** including Diebold-Mariano tests
- **Baseline model comparisons** (naive, moving average, random walk)
- **Enhanced research data** in CSV format with statistical metrics
- **Reproducible methodology** with shared utility functions

## Technical Implementation Details

### Data Sources
- **VIX Data**: Yahoo Finance (^VIX) - CBOE Volatility Index
- **VVIX Data**: Yahoo Finance (^VVIX) - VIX of VIX for volatility-of-volatility
- **Time Period**: Historical data through May 2025, evaluation from June 2025-present

### Feature Engineering
- **Technical Indicators**: Moving averages, RSI, Bollinger Bands, volatility measures
- **Lagged Features**: Multiple time lags for temporal dependencies
- **Statistical Features**: Rolling statistics, z-scores, percentile ranks
- **Dimensionality Reduction**: PCA optimization for feature selection

### Model Architectures

#### CNN-LSTM Variants
- **Basic**: Standard CNN + LSTM layers
- **Deep**: Multiple CNN and LSTM layers with batch normalization
- **Attention**: Multi-head attention mechanisms (8+4 heads)
- **Bidirectional**: Bidirectional LSTM for forward/backward processing
- **Multi-scale**: Multiple CNN branches with different kernel sizes

#### GRU Variants
- **Basic**: Standard GRU layers
- **Deep**: Stacked GRU layers with regularization
- **Attention**: Self-attention mechanisms (6 heads)
- **Bidirectional**: Bidirectional GRU processing
- **Residual**: Skip connections for gradient flow

### Optimization Framework
- **Hyperparameter Optimization**: Optuna framework with Tree-structured Parzen Estimator
- **Objective Function**: Minimize validation loss on next-day VIX prediction
- **Cross-validation**: Time series split for temporal data integrity
- **Early Stopping**: Prevent overfitting with patience-based stopping

## Research Output

### Generated Results
- **VIX_Model_Results_and_Errors_from_June2025.csv**: Complete prediction results and error analysis
- **Hyperparameter optimization logs**: Best parameters for each model architecture
- **Performance visualizations**: Training curves, prediction accuracy plots, error distributions
- **Statistical comparison**: Model performance metrics and significance tests

### CSV File Structure
```
Date, Actual_VIX, CNN_LSTM_Prediction, GRU_Prediction, GARCH_Prediction,
Ensemble_Prediction, CNN_LSTM_Error, GRU_Error, GARCH_Error,
Best_Model, Absolute_Errors, Squared_Errors, Percentage_Errors
```

## Research Applications

### Academic Research
- **Volatility Forecasting Studies**: Benchmark modern ML against traditional econometric methods
- **Model Architecture Analysis**: Systematic comparison of neural network designs
- **Financial Time Series**: Advanced techniques for financial market prediction
- **Attention Mechanisms**: Application of attention in financial forecasting

### Practical Applications
- **Risk Management**: Enhanced volatility prediction for portfolio management
- **Derivatives Pricing**: Improved VIX forecasts for options and volatility products
- **Market Analysis**: Understanding volatility regime changes and market stress periods
- **Trading Strategies**: VIX-based trading signal generation

## Troubleshooting

### Memory Issues
```python
# Reduce batch size in optimization notebooks
batch_size = 8  # Instead of 16 or 32
```

### Slow Training
```python
# Reduce optimization trials
n_trials = 50  # Instead of 100
epochs = 100   # Instead of 200
```

### GPU Issues
```python
# Force CPU usage if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Dependencies

### Core Requirements
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
yfinance>=0.1.87
arch>=5.3.0
optuna>=3.0.0
jupyter>=1.0.0
```

### Installation Notes
- **TensorFlow**: GPU support optional but recommended for faster training
- **Optuna**: Required for hyperparameter optimization
- **ARCH**: Essential for GARCH model implementation
- **YFinance**: For real-time financial data download

## Research Methodology References

### VIX Forecasting Literature
- **Duan (1995)**: GARCH methodology for volatility modeling
- **Poon & Granger (2003)**: Forecasting volatility of asset returns
- **Andersen et al. (2006)**: Realized volatility forecasting

### Deep Learning for Finance
- **Fischer & Krauss (2018)**: Deep learning with long short-term memory networks
- **Sezer et al. (2020)**: Financial time series forecasting with deep learning
- **Jiang (2021)**: Applications of deep learning in finance

### Attention Mechanisms
- **Vaswani et al. (2017)**: Attention is all you need
- **Li et al. (2019)**: Enhancing time series forecasting with attention
- **Zhou et al. (2021)**: Informer: Beyond efficient transformer for long sequence time-series forecasting

## Citation

If you use this research in your work, please cite:
```
VIX Forecasting Research Project: Assessing Modern Machine Learning Capabilities
for Next-Day Volatility Prediction. [Year]. Available at: [Repository URL]
```

## Contact

For research collaboration or questions about methodology, please refer to the notebook documentation and implementation details provided in each analysis file.

---

**Research Focus**: Advancing the understanding of machine learning applications in financial volatility forecasting through systematic empirical analysis.
