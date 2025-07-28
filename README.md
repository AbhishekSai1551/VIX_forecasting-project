# VIX Forecasting with CNN-LSTM and GRU Models

## Overview

This project implements a comprehensive comparison of CNN-LSTM and GRU architectures for predicting tomorrow's VIX (Volatility Index) values. The research systematically evaluates 11 different neural network architectures against traditional GARCH models to identify the optimal approach for VIX forecasting.

## Key Features

- **Multiple Architecture Testing**: 5 CNN-LSTM variants + 6 GRU variants
- **Hyperparameter Optimization**: Optuna-based optimization for each architecture
- **Statistical Validation**: Diebold-Mariano tests and time series cross-validation
- **GARCH Baseline**: Implementation of Duan's GARCH methodology for comparison
- **Comprehensive Analysis**: Detailed performance metrics and visualizations
- **May 2025+ Focus**: Evaluation period from May 2025 onwards

## Architecture Variants

### CNN-LSTM Models (5 variants)
1. **Basic CNN-LSTM**: Standard CNN + LSTM layers
2. **Deep CNN-LSTM**: Multiple CNN and LSTM layers with batch normalization
3. **Bidirectional CNN-LSTM**: Bidirectional LSTM for better temporal modeling
4. **Attention CNN-LSTM**: Multi-head attention mechanism
5. **Multiscale CNN-LSTM**: Multiple kernel sizes for multi-scale feature extraction

### GRU Models (6 variants)
1. **Basic GRU**: Standard GRU layers
2. **Deep GRU**: Stacked GRU layers with regularization
3. **Bidirectional GRU**: Bidirectional processing
4. **Attention GRU**: Self-attention mechanisms
5. **Residual GRU**: Skip connections for gradient flow
6. **Dropout-Enhanced GRU**: Advanced regularization strategies

## Project Structure

```
VIX_forecasting_project/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── vix_research_utils.py                       # Shared utility functions
├── CNN_LSTM_Architecture_Testing.ipynb         # CNN-LSTM model testing
├── GRU_Architecture_Testing.ipynb              # GRU model testing
├── VIX_Final_Predictions.ipynb                 # Final model comparison
├── VIX_Forecasting_Restructure_Plan.md         # Project methodology
└── VIX_Research_Findings_and_Recommendations.md # Research findings
```

## Installation

### Prerequisites
- Python 3.8+ (recommended: Python 3.8-3.10)
- 8GB+ RAM for model training
- GPU recommended (optional, for faster training)

### Setup Instructions

1. **Clone or download the project files**

2. **Create and activate virtual environment**:
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How to Run the Application

### Step 1: CNN-LSTM Architecture Testing
```bash
jupyter notebook CNN_LSTM_Architecture_Testing.ipynb
```
- Tests 5 different CNN-LSTM architectures
- Performs hyperparameter optimization (50 trials per architecture)
- Saves results to `cnn_lstm_comprehensive_results.pkl`
- **Expected runtime**: 2-4 hours

### Step 2: GRU Architecture Testing
```bash
jupyter notebook GRU_Architecture_Testing.ipynb
```
- Tests 6 different GRU architectures
- Performs hyperparameter optimization (50 trials per architecture)
- Saves results to `gru_comprehensive_results.pkl`
- **Expected runtime**: 2-4 hours

### Step 3: Final Model Comparison
```bash
jupyter notebook VIX_Final_Predictions.ipynb
```
- Loads best models from previous steps
- Implements Duan's GARCH model for comparison
- Generates predictions for May 2025+ period
- Creates comprehensive visualizations and analysis
- Saves results to `VIX_Model_Results_and_Errors_from_May2025.csv`
- **Expected runtime**: 30-60 minutes

## Quick Start (Automated)

If you want to run all notebooks sequentially:

```bash
# Start Jupyter
jupyter notebook

# Then run notebooks in this order:
# 1. CNN_LSTM_Architecture_Testing.ipynb
# 2. GRU_Architecture_Testing.ipynb  
# 3. VIX_Final_Predictions.ipynb
```

## Data Sources

- **VIX Data**: Yahoo Finance (^VIX) - CBOE Volatility Index
- **VVIX Data**: Yahoo Finance (^VVIX) - VIX of VIX (volatility-of-volatility)
- **Time Period**: Historical data from 2010, evaluation from May 2025 onwards
- **Features**: Technical indicators, lagged values, GARCH volatility, moving averages

## Expected Results

### Performance Targets
- **Best CNN-LSTM**: R² = 0.75-0.85, Directional Accuracy = 70-75%
- **Best GRU**: R² = 0.70-0.80, Directional Accuracy = 65-70%
- **GARCH Baseline**: R² = 0.60-0.70, Directional Accuracy = 60-65%
- **Ensemble Model**: R² = 0.80-0.90, Directional Accuracy = 75-80%

### Output Files
- `cnn_lstm_comprehensive_results.pkl`: CNN-LSTM testing results
- `gru_comprehensive_results.pkl`: GRU testing results
- `VIX_Model_Results_and_Errors_from_May2025.csv`: Final predictions and errors
- `VIX_Model_Performance_Summary.csv`: Performance comparison table

## Key Research Findings

1. **CNN-LSTM architectures generally outperform GRU architectures** for VIX forecasting
2. **Ensemble methods provide the best overall performance** by combining model strengths
3. **Optimal hyperparameters**: Learning rates 1e-4 to 1e-3, dropout 0.2-0.4, 30-day sequences
4. **Feature importance**: Lagged VIX values, VVIX data, and GARCH volatility are most predictive
5. **Statistical significance**: Deep learning models show significant improvements over traditional approaches

## Troubleshooting

### Memory Issues
```python
# Reduce batch size in notebooks
batch_size = 16  # Instead of 32
```

### Slow Training
```python
# Reduce optimization trials
n_trials = 25  # Instead of 50
```

### GPU Issues
```python
# Force CPU usage if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Research Applications

### Academic Research
- Volatility forecasting methodology comparison
- Modern ML vs traditional econometric approaches
- Neural network architecture analysis for financial time series

### Practical Applications
- Risk management and portfolio optimization
- VIX-based trading strategies
- Derivatives pricing and volatility surface modeling
- Market stress testing and scenario analysis

## Documentation

- **[VIX_Forecasting_Restructure_Plan.md](VIX_Forecasting_Restructure_Plan.md)**: Detailed project methodology
- **[VIX_Research_Findings_and_Recommendations.md](VIX_Research_Findings_and_Recommendations.md)**: Comprehensive research findings and recommendations

## Dependencies

Core requirements:
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing
- `tensorflow>=2.10.0`: Deep learning framework
- `optuna>=3.0.0`: Hyperparameter optimization
- `arch>=5.3.0`: GARCH modeling
- `yfinance>=0.1.87`: Financial data download
- `scikit-learn>=1.1.0`: Machine learning utilities
- `matplotlib>=3.5.0`: Visualization
- `seaborn>=0.11.0`: Statistical visualization

## Contact & Support

For questions about methodology, implementation, or research applications, please refer to the comprehensive documentation provided in the project files.

---

**Research Focus**: Advancing VIX forecasting through systematic comparison of modern deep learning architectures with rigorous statistical validation.
