# Power Load Forecasting Model

A comprehensive deep learning model for power load time series forecasting that integrates multiple advanced components for accurate multi-step predictions.

## Overview

This model is specifically designed for electrical power load forecasting and incorporates:

- **Multi-scale temporal feature extraction** - Captures patterns at different time scales (short, medium, long-term)
- **Frequency domain processing** - Analyzes periodic patterns using FFT-based transformations
- **Wavelet-based feature transformation** - Multi-resolution analysis of time series
- **Cross-attention mechanisms** - Adaptive feature fusion between different representations
- **Support for multivariate inputs** - Handles load, temperature, humidity, and other factors
- **Periodicity and seasonality handling** - Captures daily, weekly, and seasonal patterns
- **Multi-step ahead prediction** - Forecasts multiple time steps into the future

## Model Architecture

```
Input (historical data)
    ↓
Input Embedding + Positional Encoding
    ↓
Multi-scale Views (short/mid/long-term)
    ↓
Multi-scale Feature Extraction (ImprovedAdaptiveBasisSelection)
    ↓
Frequency Domain Processing (FreMLP_bottle)
    ↓
Wavelet Transformation (AdpWaveletBlock)
    ↓
Feature Fusion (SqueezeAndExciteFusionAdd1D)
    ↓
Series Encoding (Multi-layer Transformer-like)
    ↓
Cross-Attention (DWCA - Dual-Way Cross Attention)
    ↓
Temporal Projection (seq_len → pred_len)
    ↓
Prediction Head
    ↓
Output (forecasted load)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- PyWavelets

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install PyWavelets numpy
```

## Quick Start

### Basic Usage

```python
import torch
from power_load_model import PowerLoadConfig, PowerLoadForecastingModel

# 1. Create model configuration
config = PowerLoadConfig(
    seq_len=96,      # 4 days of hourly data (input)
    pred_len=24,     # 1 day ahead prediction (output)
    enc_in=7,        # Number of input features
    d_model=512,     # Model dimension
    n_heads=8,       # Attention heads
    basis_nums=32    # Basis functions
)

# 2. Initialize model
model = PowerLoadForecastingModel(config)

# 3. Prepare your data
# Shape: [batch_size, seq_len, num_features]
# Features could be: [load, temperature, humidity, wind_speed, ...]
input_data = torch.randn(32, 96, 7)

# 4. Forward pass
predictions = model(input_data)
# Output shape: [batch_size, pred_len, 1]
print(predictions.shape)  # torch.Size([32, 24, 1])
```

### Training Example

```python
import torch.optim as optim

# Initialize model
config = PowerLoadConfig(seq_len=96, pred_len=24, enc_in=7)
model = PowerLoadForecastingModel(config)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # batch_x: [B, seq_len, enc_in]
        # batch_y: [B, pred_len, 1]
        
        # Forward pass
        predictions = model(batch_x)
        
        # Compute loss
        loss_dict = model.compute_loss(predictions, batch_y)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Inference Example

```python
# Load trained model
model.eval()

# Make predictions
with torch.no_grad():
    test_data = torch.randn(10, 96, 7)
    predictions = model.predict(test_data)
    
print(f"Predictions shape: {predictions.shape}")  # (10, 24)
```

### Model Save/Load

```python
# Save model
model.save_model('power_load_model.pth')

# Load model
loaded_model = PowerLoadForecastingModel.load_model(
    'power_load_model.pth',
    device='cpu'
)
```

## Configuration Parameters

### PowerLoadConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | 96 | Input sequence length (historical time steps) |
| `pred_len` | int | 24 | Prediction horizon (future time steps) |
| `enc_in` | int | 7 | Number of input features |
| `d_model` | int | 512 | Model hidden dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `basis_nums` | int | 32 | Number of basis functions |
| `dropout` | float | 0.1 | Dropout rate |
| `freq_bottleneck` | int | 256 | Frequency domain bottleneck dimension |
| `lifting_kernel_size` | int | 4 | Wavelet transform kernel size |
| `regu_details` | float | 0.1 | Wavelet detail regularization |
| `regu_approx` | float | 0.1 | Wavelet approximation regularization |
| `use_holiday_features` | bool | True | Use external/holiday features |
| `multi_scale_layers` | int | 3 | Number of multi-scale layers |

## Input Data Format

### Input Features (batch_x)

Shape: `[batch_size, seq_len, enc_in]`

Example features for power load forecasting:
1. Historical load values
2. Temperature
3. Humidity  
4. Wind speed
5. Solar radiation
6. Day of week (encoded)
7. Hour of day (encoded)

### Target (batch_y)

Shape: `[batch_size, pred_len, 1]`

Future power load values to predict.

## Advanced Features

### Multi-Scale Processing

The model automatically creates three temporal views:
- **Short-term**: Full resolution (captures immediate patterns)
- **Mid-term**: 2x downsampled (captures medium-range trends)
- **Long-term**: 4x downsampled (captures seasonal patterns)

### Attention Visualization

```python
# Get attention weights for analysis
predictions, attention_dict = model(input_data, return_attention=True)

# Access attention information
basis_attention = attention_dict['attn_basis']
series_attention = attention_dict['attn_series']
coefficients = attention_dict['coef']
wavelet_details = attention_dict['wave_detail']
```

### External Features

```python
# Include holiday or special event features
external_features = torch.randn(batch_size, seq_len, d_model)
predictions = model(input_data, external_features=external_features)
```

### Custom Loss Function

```python
# Compute detailed loss information
predictions = model(input_data)
loss_dict = model.compute_loss(predictions, targets)

# Access individual loss components
total_loss = loss_dict['total_loss']    # Combined loss
mse_loss = loss_dict['mse_loss']        # Mean squared error
mae_loss = loss_dict['mae_loss']        # Mean absolute error
freq_loss = loss_dict['freq_loss']      # Frequency regularization
```

## Model Components

### 1. ImprovedAdaptiveBasisSelection
Extracts multi-scale features and generates adaptive basis functions for representing the time series.

### 2. FreMLP_bottle
Processes features in the frequency domain using FFT, capturing periodic patterns efficiently.

### 3. AdpWaveletBlock
Applies wavelet transformation for multi-resolution analysis, separating low and high-frequency components.

### 4. DWCA (Dual-Way Cross Attention)
Implements bidirectional cross-attention between basis functions and series features.

### 5. SqueezeAndExciteFusionAdd1D
Fuses features from different sources using channel attention mechanisms.

## Performance Considerations

### Memory Usage

The model size depends on configuration:
- Default config (~512 dim, 8 heads): ~32M parameters
- Smaller config (256 dim, 4 heads): ~8M parameters
- Larger config (1024 dim, 16 heads): ~130M parameters

### GPU Recommendations

- Training: NVIDIA GPU with 8GB+ VRAM
- Inference: Can run on CPU for smaller batches
- Recommended batch sizes:
  - GPU (8GB): 64-128
  - GPU (16GB): 128-256
  - CPU: 8-32

### Speed Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    predictions = model(input_data)
    loss = model.compute_loss(predictions, targets)['total_loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Example Applications

### 1. Day-Ahead Load Forecasting

```python
# Configure for 24-hour ahead prediction
config = PowerLoadConfig(
    seq_len=168,      # 1 week of hourly data
    pred_len=24,      # Next 24 hours
    enc_in=10         # Multiple weather and calendar features
)
```

### 2. Week-Ahead Load Forecasting

```python
# Configure for weekly prediction
config = PowerLoadConfig(
    seq_len=672,      # 4 weeks of hourly data
    pred_len=168,     # Next week (168 hours)
    enc_in=7
)
```

### 3. Real-Time Forecasting

```python
# Configure for short-term prediction
config = PowerLoadConfig(
    seq_len=48,       # 2 days of hourly data
    pred_len=6,       # Next 6 hours
    enc_in=5
)
```

## Testing

Run the included test suite:

```bash
python power_load_model.py
```

This will run:
1. Model configuration tests
2. Forward pass tests with different batch sizes
3. Loss computation tests
4. Model save/load tests
5. Component integration tests

## Troubleshooting

### Common Issues

**Issue**: Out of memory error
- **Solution**: Reduce `batch_size`, `d_model`, or `seq_len`

**Issue**: NaN in loss
- **Solution**: Reduce learning rate, check input normalization

**Issue**: Slow training
- **Solution**: Use GPU, reduce `basis_nums` or `multi_scale_layers`

**Issue**: Poor predictions
- **Solution**: Increase `seq_len` for more context, normalize input features, tune hyperparameters

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{fwbnet_power_load_2025,
  title={Power Load Forecasting Model with Multi-Scale Attention},
  author={fwbnet},
  year={2025},
  url={https://github.com/You0610/fwbnet}
}
```

## License

This project is part of the fwbnet repository. Please refer to the repository's license.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

This model integrates several advanced time series forecasting techniques including:
- Frequency domain processing
- Wavelet transformation
- Multi-head attention mechanisms
- Squeeze-and-excitation networks
- Adaptive basis selection
