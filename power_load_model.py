"""
Power Load Forecasting Model

This module implements a comprehensive power load time series forecasting model
that integrates multiple advanced components for multi-scale feature extraction,
frequency domain processing, and adaptive attention mechanisms.

Key Features:
- Multi-scale temporal feature extraction
- Frequency and time domain dual processing
- Wavelet-based feature transformation
- Cross-attention mechanisms for adaptive feature fusion
- Support for multivariate inputs (load, temperature, humidity, etc.)
- Handles periodicity and seasonality in power load data
- Multi-step ahead prediction

Author: fwbnet
Date: 2024-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

# Import all necessary modules from utils_y.py
from utils_y import (
    ImprovedAdaptiveBasisSelection,
    FreMLP_bottle,
    DWCA,
    AdpWaveletBlock,
    SqueezeAndExciteFusionAdd1D,
    Config
)


class PowerLoadConfig:
    """
    Configuration class for Power Load Forecasting Model.
    
    This class manages all hyperparameters and settings for the power load
    forecasting model, including data dimensions, model architecture parameters,
    and training configurations.
    
    Attributes:
        seq_len (int): Input sequence length (historical time steps)
        pred_len (int): Prediction horizon (future time steps to predict)
        enc_in (int): Number of input features/variables
        d_model (int): Model hidden dimension
        n_heads (int): Number of attention heads
        basis_nums (int): Number of basis functions for adaptive selection
        dropout (float): Dropout rate for regularization
        freq_bottleneck (int): Bottleneck dimension for frequency processing
        lifting_kernel_size (int): Kernel size for wavelet lifting scheme
        regu_details (float): Regularization weight for wavelet details
        regu_approx (float): Regularization weight for wavelet approximation
        use_holiday_features (bool): Whether to include holiday features
        multi_scale_layers (int): Number of multi-scale processing layers
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        enc_in: int = 7,
        d_model: int = 512,
        n_heads: int = 8,
        basis_nums: int = 32,
        dropout: float = 0.1,
        freq_bottleneck: int = 256,
        lifting_kernel_size: int = 4,
        regu_details: float = 0.1,
        regu_approx: float = 0.1,
        use_holiday_features: bool = True,
        multi_scale_layers: int = 3
    ):
        """
        Initialize PowerLoadConfig with model hyperparameters.
        
        Args:
            seq_len: Length of input sequence (e.g., 96 for 4 days of hourly data)
            pred_len: Length of prediction horizon (e.g., 24 for 1 day ahead)
            enc_in: Number of input features (load, temperature, humidity, etc.)
            d_model: Hidden dimension of the model
            n_heads: Number of attention heads
            basis_nums: Number of basis functions
            dropout: Dropout probability
            freq_bottleneck: Dimension for frequency domain bottleneck
            lifting_kernel_size: Kernel size for wavelet transform
            regu_details: Regularization for wavelet high-frequency components
            regu_approx: Regularization for wavelet low-frequency components
            use_holiday_features: Whether to use holiday/external features
            multi_scale_layers: Number of layers for multi-scale processing
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.basis_nums = basis_nums
        self.dropout = dropout
        self.freq_bottleneck = freq_bottleneck
        self.lifting_kernel_size = lifting_kernel_size
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.use_holiday_features = use_holiday_features
        self.multi_scale_layers = multi_scale_layers
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.pred_len > 0, "pred_len must be positive"
        assert self.enc_in > 0, "enc_in must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.basis_nums > 0, "basis_nums must be positive"
        assert self.freq_bottleneck > 0, "freq_bottleneck must be positive"
        assert self.multi_scale_layers > 0, "multi_scale_layers must be positive"
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'basis_nums': self.basis_nums,
            'dropout': self.dropout,
            'freq_bottleneck': self.freq_bottleneck,
            'lifting_kernel_size': self.lifting_kernel_size,
            'regu_details': self.regu_details,
            'regu_approx': self.regu_approx,
            'use_holiday_features': self.use_holiday_features,
            'multi_scale_layers': self.multi_scale_layers
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create configuration from dictionary."""
        return cls(**config_dict)


class PowerLoadForecastingModel(nn.Module):
    """
    Power Load Forecasting Model for Time Series Prediction.
    
    This model integrates multiple advanced components to perform accurate
    multi-step power load forecasting. It processes multi-scale temporal features,
    applies frequency domain transformations, and uses adaptive attention mechanisms
    to capture complex patterns in power load data.
    
    Architecture:
        1. Multi-scale temporal feature extraction using ImprovedAdaptiveBasisSelection
        2. Frequency domain processing with FreMLP_bottle
        3. Wavelet-based transformation using AdpWaveletBlock
        4. Dual-way cross attention with DWCA
        5. Feature fusion with SqueezeAndExciteFusionAdd1D
        6. Final prediction layer
    
    The model is designed to handle:
        - Multivariate inputs (load, temperature, humidity, etc.)
        - Periodicity and seasonality in power consumption
        - Holiday and external factors
        - Multi-step ahead predictions
    
    Args:
        config (PowerLoadConfig): Model configuration object
    
    Example:
        >>> config = PowerLoadConfig(seq_len=96, pred_len=24, enc_in=7)
        >>> model = PowerLoadForecastingModel(config)
        >>> x = torch.randn(32, 96, 7)  # batch_size=32, seq_len=96, features=7
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 24, 1])
    """
    
    def __init__(self, config: PowerLoadConfig):
        """
        Initialize the Power Load Forecasting Model.
        
        Args:
            config: PowerLoadConfig object containing model hyperparameters
        """
        super(PowerLoadForecastingModel, self).__init__()
        
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.d_model = config.d_model
        
        # Input embedding layer - projects input features to model dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(config.enc_in, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(config.d_model, config.seq_len)
        
        # Multi-scale feature extraction
        # Creates short-term, mid-term, and long-term views of the data
        self.multi_scale_extractor = ImprovedAdaptiveBasisSelection(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            basis_nums=config.basis_nums,
            heads=config.n_heads
        )
        
        # Frequency domain processing
        # Captures periodic patterns in frequency space
        self.freq_processor = FreMLP_bottle(
            input_len=config.d_model,
            output_len=config.d_model,
            bottleneck=config.freq_bottleneck,
            bias=True
        )
        
        # Wavelet transform configuration
        wavelet_config = Config()
        wavelet_config.enc_in = config.d_model
        wavelet_config.lifting_kernel_size = config.lifting_kernel_size
        wavelet_config.regu_details = config.regu_details
        wavelet_config.regu_approx = config.regu_approx
        
        # Wavelet-based multi-resolution analysis
        self.wavelet_transform = AdpWaveletBlock(wavelet_config, config.d_model)
        
        # Dual-way cross attention for adaptive feature interaction
        self.cross_attention = DWCA(
            d_model=config.d_model,
            heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Feature fusion module
        self.feature_fusion = SqueezeAndExciteFusionAdd1D(config.d_model)
        
        # Series encoder - processes the main time series
        self.series_encoder = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model * 4, config.d_model),
                nn.Dropout(config.dropout)
            ) for _ in range(config.multi_scale_layers)
        ])
        
        # Projection to prediction horizon
        self.temporal_projection = nn.Sequential(
            nn.Linear(config.seq_len, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)  # Predict single value (load)
        )
        
        # Optional: Holiday/external feature processor
        if config.use_holiday_features:
            self.holiday_processor = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_multi_scale_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create multi-scale temporal views of input data.
        
        This method generates three different temporal scales:
        - Short-term: Recent detailed patterns
        - Mid-term: Medium-range trends
        - Long-term: Long-range seasonal patterns
        
        Args:
            x: Input tensor of shape [B, L, C]
        
        Returns:
            Tuple of (short_term, mid_term, long_term) tensors
        """
        B, L, C = x.shape
        
        # Short-term: Use full resolution (all time steps)
        short_term = x  # [B, L, C]
        
        # Mid-term: Downsample by factor of 2
        if L >= 4:
            mid_term = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            # Upsample back to original length
            mid_term = F.interpolate(mid_term.transpose(1, 2), size=L, mode='linear', align_corners=True).transpose(1, 2)
        else:
            mid_term = x
        
        # Long-term: Downsample by factor of 4
        if L >= 8:
            long_term = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
            # Upsample back to original length
            long_term = F.interpolate(long_term.transpose(1, 2), size=L, mode='linear', align_corners=True).transpose(1, 2)
        else:
            long_term = x
        
        return short_term, mid_term, long_term
    
    def forward(
        self, 
        x: torch.Tensor,
        external_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the power load forecasting model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, enc_in]
               Contains historical power load and related features
            external_features: Optional tensor of shape [batch_size, seq_len, d_model]
                             Pre-embedded external features like holidays, special events.
                             Must be already embedded to d_model dimension.
                             If shape is [batch_size, seq_len, enc_in], it will be embedded.
            return_attention: Whether to return attention weights for visualization
        
        Returns:
            predictions: Tensor of shape [batch_size, pred_len, 1]
                        Predicted power load values
        
        Note:
            The model processes the input through multiple stages:
            1. Embedding and positional encoding
            2. Multi-scale feature extraction
            3. Frequency domain processing
            4. Wavelet transformation
            5. Cross-attention based fusion
            6. Temporal projection and prediction
        """
        B, L, C = x.shape
        
        # 1. Input embedding and positional encoding
        x_embed = self.input_embedding(x)  # [B, L, d_model]
        x_embed = self.positional_encoding(x_embed)  # [B, L, d_model]
        
        # 2. Create multi-scale temporal views
        short_term, mid_term, long_term = self._create_multi_scale_views(x_embed)
        
        # 3. Multi-scale feature extraction
        # Extract basis functions and projection features from multi-scale inputs
        basis_features, proj_features = self.multi_scale_extractor(
            short_term, mid_term, long_term, x_embed
        )  # basis_features: [B, N, d_model], proj_features: [B, 128]
        
        # 4. Frequency domain processing
        # Process each time step through frequency domain
        x_freq = []
        for i in range(L):
            freq_out = self.freq_processor(x_embed[:, i, :])  # [B, d_model]
            x_freq.append(freq_out)
        x_freq = torch.stack(x_freq, dim=1)  # [B, L, d_model]
        
        # 5. Wavelet transformation
        # Apply wavelet transform for multi-resolution analysis
        x_wave_input = x_freq.transpose(1, 2)  # [B, d_model, L]
        x_wave, wave_reg, wave_detail = self.wavelet_transform(x_wave_input)
        x_wave = x_wave.transpose(1, 2)  # [B, L, d_model]
        
        # 6. Feature fusion (wavelet + frequency)
        x_fused = self.feature_fusion(x_freq, x_wave)  # [B, L, d_model]
        
        # 7. Series encoding with residual connections
        x_series = x_fused
        for encoder_layer in self.series_encoder:
            x_series = x_series + encoder_layer(x_series)
        
        # 8. Cross-attention between basis and series features
        # This adaptively combines learned basis functions with encoded series
        coef, attn_basis, attn_series = self.cross_attention(
            basis_features, x_series
        )  # coef: [B, H, L, N]
        
        # 9. Apply coefficients to reconstruct enhanced features
        # Aggregate information across attention heads
        coef_mean = coef.mean(dim=1)  # [B, L, N]
        
        # Weighted combination of basis functions
        enhanced_features = torch.matmul(coef_mean, basis_features)  # [B, L, d_model]
        
        # 10. Optional: Incorporate external features
        if external_features is not None and self.config.use_holiday_features:
            # Check if external_features need embedding
            if external_features.shape[-1] == self.config.enc_in:
                # External features have same dimension as input, need embedding
                ext_embed = self.input_embedding(external_features)  # [B, L, d_model]
            elif external_features.shape[-1] == self.config.d_model:
                # External features are already embedded to d_model
                ext_embed = external_features
            else:
                raise ValueError(
                    f"external_features last dimension must be either {self.config.enc_in} "
                    f"or {self.config.d_model}, got {external_features.shape[-1]}"
                )
            ext_processed = self.holiday_processor(ext_embed)
            enhanced_features = enhanced_features + ext_processed
        
        # 11. Temporal projection to prediction horizon
        # Transform from seq_len to pred_len
        enhanced_features = enhanced_features.transpose(1, 2)  # [B, d_model, L]
        projected = self.temporal_projection(enhanced_features)  # [B, d_model, pred_len]
        projected = projected.transpose(1, 2)  # [B, pred_len, d_model]
        
        # 12. Final prediction
        predictions = self.prediction_head(projected)  # [B, pred_len, 1]
        
        if return_attention:
            return predictions, {
                'attn_basis': attn_basis,
                'attn_series': attn_series,
                'coef': coef,
                'wave_detail': wave_detail
            }
        
        return predictions
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with multiple components.
        
        Args:
            predictions: Model predictions [B, pred_len, 1]
            targets: Ground truth values [B, pred_len, 1]
            reduction: Loss reduction method ('mean', 'sum', 'none')
        
        Returns:
            Dictionary containing:
                - 'total_loss': Combined loss
                - 'mse_loss': Mean squared error
                - 'mae_loss': Mean absolute error
                - 'freq_loss': Frequency domain loss (if available)
        """
        # Main prediction loss (MSE)
        mse_loss = F.mse_loss(predictions, targets, reduction=reduction)
        
        # MAE loss for robustness
        mae_loss = F.l1_loss(predictions, targets, reduction=reduction)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * mae_loss
        
        # Add frequency domain loss if available
        freq_loss = torch.tensor(0.0, device=predictions.device)
        if hasattr(self.freq_processor, 'spec_loss'):
            freq_loss = self.freq_processor.spec_loss
            total_loss = total_loss + 0.01 * freq_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'freq_loss': freq_loss
        }
    
    def predict(
        self, 
        x: torch.Tensor,
        external_features: Optional[torch.Tensor] = None,
        denormalize_fn: Optional[callable] = None
    ) -> np.ndarray:
        """
        Make predictions in inference mode.
        
        Args:
            x: Input tensor [B, seq_len, enc_in]
            external_features: Optional external features
            denormalize_fn: Optional function to denormalize predictions
        
        Returns:
            Numpy array of predictions [B, pred_len]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, external_features)
            predictions = predictions.squeeze(-1)  # [B, pred_len]
            
            if denormalize_fn is not None:
                predictions = denormalize_fn(predictions)
            
            return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """
        Save model state and configuration.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to model checkpoint
            device: Device to load model on ('cpu' or 'cuda')
        
        Returns:
            Loaded PowerLoadForecastingModel instance
        """
        checkpoint = torch.load(path, map_location=device)
        config = PowerLoadConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding temporal position information.
    
    Uses sinusoidal position encodings as described in "Attention is All You Need".
    This helps the model understand the temporal ordering of the input sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, L, d_model]
        
        Returns:
            Tensor with positional encoding added [B, L, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


def create_sample_data(
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 24,
    num_features: int = 7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample data for testing the model.
    
    This function generates synthetic power load data with realistic patterns:
    - Daily periodicity
    - Weekly seasonality
    - Random noise
    
    Args:
        batch_size: Number of samples in batch
        seq_len: Length of input sequence
        pred_len: Length of prediction horizon
        num_features: Number of input features
    
    Returns:
        Tuple of (input_data, target_data)
    """
    # Create time indices
    total_len = seq_len + pred_len
    t = torch.linspace(0, total_len / 24, total_len)  # Time in days
    
    # Generate synthetic power load with daily and weekly patterns
    daily_pattern = torch.sin(2 * np.pi * t)  # Daily cycle
    weekly_pattern = 0.3 * torch.sin(2 * np.pi * t / 7)  # Weekly cycle
    trend = 0.1 * t  # Slight upward trend
    noise = 0.1 * torch.randn(total_len)
    
    # Combine patterns
    base_load = 50 + 20 * (daily_pattern + weekly_pattern) + trend + noise
    
    # Create batch
    input_data = torch.zeros(batch_size, seq_len, num_features)
    target_data = torch.zeros(batch_size, pred_len, 1)
    
    for i in range(batch_size):
        # Slight variation for each sample
        variation = 1 + 0.1 * torch.randn(1).item()
        load_series = base_load * variation
        
        # Input features
        input_data[i, :, 0] = load_series[:seq_len]  # Load
        input_data[i, :, 1] = 20 + 5 * torch.sin(2 * np.pi * t[:seq_len]) + torch.randn(seq_len)  # Temperature
        input_data[i, :, 2] = 60 + 10 * torch.randn(seq_len)  # Humidity
        input_data[i, :, 3:] = torch.randn(seq_len, num_features - 3) * 0.5  # Other features
        
        # Target (future load)
        target_data[i, :, 0] = load_series[seq_len:seq_len + pred_len]
    
    return input_data, target_data


# ============================================================================
# Usage Example and Testing Code
# ============================================================================

def example_usage():
    """
    Demonstrate basic usage of the PowerLoadForecastingModel.
    
    This example shows:
    1. Model configuration
    2. Model initialization
    3. Training step
    4. Inference/prediction
    5. Model save/load
    """
    print("=" * 80)
    print("Power Load Forecasting Model - Usage Example")
    print("=" * 80)
    
    # 1. Configure the model
    print("\n1. Creating model configuration...")
    config = PowerLoadConfig(
        seq_len=96,          # 4 days of hourly data
        pred_len=24,         # Predict 1 day ahead
        enc_in=7,            # 7 input features (load, temp, humidity, etc.)
        d_model=512,         # Model dimension
        n_heads=8,           # Number of attention heads
        basis_nums=32,       # Number of basis functions
        dropout=0.1,
        freq_bottleneck=256,
        use_holiday_features=True,
        multi_scale_layers=3
    )
    print(f"Configuration: {config.to_dict()}")
    
    # 2. Create model instance
    print("\n2. Initializing model...")
    model = PowerLoadForecastingModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Create sample data
    print("\n3. Generating sample data...")
    input_data, target_data = create_sample_data(
        batch_size=32,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        num_features=config.enc_in
    )
    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target_data.shape}")
    
    # 4. Training mode - forward pass and loss computation
    print("\n4. Training mode demonstration...")
    model.train()
    predictions = model(input_data)
    print(f"Predictions shape: {predictions.shape}")
    
    # Compute loss
    loss_dict = model.compute_loss(predictions, target_data)
    print(f"Training losses:")
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # 5. Inference mode - make predictions
    print("\n5. Inference mode demonstration...")
    model.eval()
    with torch.no_grad():
        pred_output, attention_dict = model(input_data[:4], return_attention=True)
    print(f"Inference predictions shape: {pred_output.shape}")
    print(f"Attention information available: {list(attention_dict.keys())}")
    
    # 6. Using predict method
    print("\n6. Using predict method...")
    predictions_np = model.predict(input_data[:4])
    print(f"Predictions (numpy): {predictions_np.shape}")
    print(f"Sample prediction values: {predictions_np[0, :5]}")
    
    # 7. Model save and load
    print("\n7. Model save/load demonstration...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        model_path = tmp.name
    
    model.save_model(model_path)
    loaded_model = PowerLoadForecastingModel.load_model(model_path, device='cpu')
    
    # Verify loaded model produces same output
    with torch.no_grad():
        original_pred = model(input_data[:2])
        loaded_pred = loaded_model(input_data[:2])
        diff = torch.abs(original_pred - loaded_pred).max().item()
    print(f"Max difference between original and loaded model: {diff:.6f}")
    
    # Cleanup
    import os
    os.remove(model_path)
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


def test_model_components():
    """
    Test individual model components to ensure they work correctly.
    """
    print("\n" + "=" * 80)
    print("Testing Model Components")
    print("=" * 80)
    
    # Test configuration
    print("\n[Test 1] Configuration validation...")
    try:
        config = PowerLoadConfig(seq_len=96, pred_len=24, enc_in=7)
        print("✓ Valid configuration created successfully")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
    
    # Test invalid configuration
    print("\n[Test 2] Invalid configuration handling...")
    try:
        bad_config = PowerLoadConfig(seq_len=-1, pred_len=24, enc_in=7)
        print("✗ Should have raised assertion error for negative seq_len")
    except AssertionError:
        print("✓ Correctly rejected invalid configuration")
    
    # Test model initialization
    print("\n[Test 3] Model initialization...")
    try:
        config = PowerLoadConfig(seq_len=96, pred_len=24, enc_in=7, d_model=512)
        model = PowerLoadForecastingModel(config)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
    
    # Test forward pass with different batch sizes
    print("\n[Test 4] Forward pass with different batch sizes...")
    try:
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 96, 7)
            output = model(x)
            assert output.shape == (batch_size, 24, 1), f"Output shape mismatch for batch_size={batch_size}"
            print(f"✓ Batch size {batch_size}: output shape {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
    
    # Test loss computation
    print("\n[Test 5] Loss computation...")
    try:
        predictions = torch.randn(32, 24, 1)
        targets = torch.randn(32, 24, 1)
        loss_dict = model.compute_loss(predictions, targets)
        assert 'total_loss' in loss_dict
        assert 'mse_loss' in loss_dict
        print(f"✓ Loss computation successful: {list(loss_dict.keys())}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
    
    # Test predict method
    print("\n[Test 6] Predict method...")
    try:
        x = torch.randn(8, 96, 7)
        predictions = model.predict(x)
        assert predictions.shape == (8, 24)
        print(f"✓ Predict method successful: output shape {predictions.shape}")
    except Exception as e:
        print(f"✗ Predict method failed: {e}")
    
    # Test multi-scale views creation
    print("\n[Test 7] Multi-scale views creation...")
    try:
        x = torch.randn(16, 96, 512)
        short, mid, long = model._create_multi_scale_views(x)
        assert short.shape == mid.shape == long.shape == x.shape
        print(f"✓ Multi-scale views created: {short.shape}")
    except Exception as e:
        print(f"✗ Multi-scale views test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Component testing completed!")
    print("=" * 80)


if __name__ == "__main__":
    """
    Run example usage and component tests when script is executed directly.
    """
    # Run usage example
    example_usage()
    
    # Run component tests
    test_model_components()
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
