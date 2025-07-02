import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import pywt
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import concurrent.futures
from torch.cuda.amp import autocast
from typing import Optional, Tuple





class FreMLP_bottle(nn.Module):
    def __init__(self, input_len, output_len, bottleneck, bias=True):
        super().__init__()
        self.embed_size = bottleneck
        self.input_len = input_len
        self.output_len = output_len
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        # 计算FFT后的维度
        self.fft_size = bottleneck // 2 + 1

        # 调整权重矩阵维度以匹配FFT输出
        self.channel_r = nn.Parameter(self.scale * torch.randn(self.fft_size, self.fft_size))
        self.channel_i = nn.Parameter(self.scale * torch.randn(self.fft_size, self.fft_size))
        self.channel_rb = nn.Parameter(self.scale * torch.randn(self.fft_size))
        self.channel_ib = nn.Parameter(self.scale * torch.randn(self.fft_size))

        self.temporal_r = nn.Parameter(self.scale * torch.randn(self.fft_size, self.fft_size))
        self.temporal_i = nn.Parameter(self.scale * torch.randn(self.fft_size, self.fft_size))
        self.temporal_rb = nn.Parameter(self.scale * torch.randn(self.fft_size))
        self.temporal_ib = nn.Parameter(self.scale * torch.randn(self.fft_size))

        # 主要映射层
        self.input_proj = wn(nn.Linear(input_len, bottleneck, bias=bias))
        self.output_proj = wn(nn.Linear(bottleneck, output_len, bias=bias))

        # Skip connection
        self.skip_proj = wn(nn.Linear(input_len, bottleneck, bias=bias))

        # 添加频域dropout和正则化
        self.freq_dropout_real = nn.Dropout(0.1)
        self.freq_dropout_imag = nn.Dropout(0.1)
        self.spectral_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.target_norm = nn.Parameter(torch.tensor(1.0))
        self.fusion = SqueezeAndExciteFusionAdd1D(bottleneck)

        self.act = nn.ReLU()

    def FreMLP(self, x, r, i, rb, ib):
        """频域MLP的核心计算"""
        # 分离实部和虚部并进行变换
        o1_real = self.act(
            torch.matmul(x.real, r) -
            torch.matmul(x.imag, i) +
            rb
        )

        o1_imag = self.act(
            torch.matmul(x.imag, r) +
            torch.matmul(x.real, i) +
            ib
        )

        # 合并实部和虚部
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def get_spectral_reg_loss(self, x_freq):
        """计算频谱正则化损失"""
        freq_norm = torch.norm(x_freq, p=2, dim=-1)
        reg_loss = F.mse_loss(freq_norm, self.target_norm.expand_as(freq_norm))
        return reg_loss * self.spectral_reg_weight

    def forward(self, x):
        # 保存输入用于skip connection
        identity = x
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # 主路径处理
        x = self.input_proj(x)

        # 频域转换
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 应用频域dropout
        x_freq_real = self.freq_dropout_real(x_freq.real)
        x_freq_imag = self.freq_dropout_imag(x_freq.imag)
        x_freq = torch.complex(x_freq_real, x_freq_imag)

        # 计算频谱正则化损失
        self.spec_loss = self.get_spectral_reg_loss(x_freq)

        # Channel-wise FreMLP
        y_channel = self.FreMLP(x_freq, self.channel_r, self.channel_i,
                                self.channel_rb, self.channel_ib)

        # Temporal FreMLP
        y_temporal = self.FreMLP(y_channel, self.temporal_r, self.temporal_i,
                                 self.temporal_rb, self.temporal_ib)

        # 逆变换回时域
        x_time = torch.fft.irfft(y_temporal, n=self.embed_size, dim=-1, norm="ortho")

        # Skip connection
        skip = self.skip_proj(identity)

        # 为了使用fusion模块,需要确保输入维度正确
        if len(x_time.shape) == 2:  # [B, D]
            x_time = x_time.unsqueeze(1)  # [B, 1, D]
            skip = skip.unsqueeze(1)  # [B, 1, D]

        # 应用fusion
        x = self.fusion(x_time, skip)

        # 恢复维度
        x = x.squeeze(1)

        x = self.output_proj(x)

        # 如果输入是1D，去掉添加的维度
        if len(identity.shape) == 1:
            x = x.squeeze(0)

        return x
class SqueezeAndExciteFusionAdd1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        reduced_dim = max(channels // reduction, 1)

        # SE模块
        self.se = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, channels),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.LayerNorm([channels * 2]),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm([channels])
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, N, d_model] - 第一个特征输入
            x2: [B, N, d_model] - 第二个特征输入
        Returns:
            [B, N, d_model] - 融合后的特征
        """

        # 计算通道统计量
        g1 = x1.mean(dim=1)  # [B, d_model]
        g2 = x2.mean(dim=1)

        # 生成注意力权重
        attn = self.se(g1)  # [B, d_model]

        # 应用注意力
        x1_enhanced = x1 * attn.unsqueeze(1)

        # 计算融合权重
        fusion_weights = self.fusion(
            torch.cat([g1, g2], dim=-1)
        )  # [B, 2]

        # 特征融合
        output = fusion_weights[:, 0:1].unsqueeze(1) * x1_enhanced + \
                 fusion_weights[:, 1:2].unsqueeze(1) * x2 + \
                 self.gamma * (x1 + x2) + self.beta

        output = self.norm(output)

        return output
class SqueezeAndExcitation1D(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super().__init__()
        
        reduction = min(reduction, channel // 2)
        reduced_dim = max(channel // reduction, 1)
        
        self.fc = nn.Sequential(
            wn(nn.Linear(channel, reduced_dim)),
            activation,
            wn(nn.Linear(reduced_dim, channel)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (B, N, D) where B is batch size, N is number of basis, D is feature dimension
        B, N, D = x.shape
        
        # Global average pooling along the basis dimension
        weighting = x.mean(dim=1)  # (B, D)
        
        # Apply channel attention
        weighting = self.fc(weighting)  # (B, D)
        
        # Reshape for broadcasting
        weighting = weighting.unsqueeze(1)  # (B, 1, D)
        
        # Apply attention weights
        y = x * weighting
        
        return y
class ImprovedAdaptiveBasisSelection(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, basis_nums, heads=4):
        super().__init__()

        self.d_model = d_model
        self.basis_nums = basis_nums
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Feature dimension transformation
        self.feature_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm([d_model]),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Multi-head attention
        valid_heads = [h for h in range(1, heads + 1) if d_model % h == 0]
        n_heads = valid_heads[-1] if valid_heads else 1

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.1,
            batch_first=True
        )


        self.fusion = SqueezeAndExciteFusionAdd1D(d_model)

        # Basis generator
        self.basis_gen = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm([d_model * 2]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, basis_nums * d_model)
        )

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128)
        )


    def process_features(self, x):
        """Process input features to correct dimensions
        Args:
            x: [B, L, C] or [B, C, L]
        Returns:
            [B, C, d_model]
        """
        B, dim1, dim2 = x.shape

        # Handle both input formats
        if dim2 == self.seq_len:
            # Input is [B, C, L]
            x = x.transpose(1, 2)  # [B, L, C]
            C = dim1
        else:
            # Input is [B, L, C]
            C = dim2

        # Project each channel independently
        x = x.permute(0, 2, 1)  # [B, C, L]
        x_reshaped = x.reshape(B * C, self.seq_len)  # [B*C, L]
        features = self.feature_proj(x_reshaped)  # [B*C, d_model]
        features = features.reshape(B, C, self.d_model)  # [B, C, d_model]


        return features

    def forward(self, short_term, mid_term, long_term, feature):
        """
        Args:
            short_term, mid_term, long_term: [B, L, C] or [B, C, L]
            feature: [B, C, L]
        """
        # 1. Feature processing
        main_features = self.process_features(feature)  # [B, C, d_model]
        short_features = self.process_features(short_term)  # [B, C, d_model]
        mid_features = self.process_features(mid_term)
        long_features = self.process_features(long_term)

        # 2. Attention processing
        # Change from [B, C, d_model] to [B, C, d_model] for attention
        attn_out, _ = self.attention(main_features, main_features, main_features)



        # 3. Multi-scale feature fusion
        scale_features = torch.stack([
            short_features,
            mid_features,
            long_features
        ], dim=1)  # [B, 3, C, d_model]

        scale_out = scale_features.mean(dim=1)  # [B, C, d_model]

        fused = self.fusion( attn_out, scale_out)

        # 5. Generate basis functions
        fused_avg = fused.mean(dim=1)  # [B, d_model]
        basis = self.basis_gen(fused_avg)  # [B, basis_nums * d_model]
        basis_features = basis.view(-1, self.basis_nums, self.d_model)  # [B, N, d_model]

        # 6. Generate projection features
        proj_features = self.proj(fused_avg)  # [B, 128]

        return basis_features, proj_features

class Splitting(nn.Module):
    def __init__(self, channel_first):
        super(Splitting, self).__init__()
        self.channel_first = channel_first

    def forward(self, x):
        """Returns the odd and even part"""
        # 确保输入长度为偶数
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1), mode='reflect')
            
        if self.channel_first:
            return x[:, :, ::2], x[:, :, 1::2]
        else:
            return x[:, ::2, :], x[:, 1::2, :]


class LiftingScheme(nn.Module):
    def __init__(self, in_channels, input_size, modified=True, splitting=True, k_size=4, simple_lifting=True):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        kernel_size = k_size
        pad = (k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(channel_first=True)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.InstanceNorm1d(in_channels)
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.InstanceNorm1d(in_channels)
            ]
        else:
            size_hidden = 2

            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels * prev_size, in_channels * size_hidden, kernel_size=kernel_size, stride=1,
                          groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels * prev_size, in_channels * size_hidden, kernel_size=kernel_size, stride=1,
                          groups=in_channels),
                nn.Tanh()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv1d(in_channels * prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv1d(in_channels * prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        # 确保输入长度为偶数
        if x.size(-1) % 2 != 0:
            # 如果是奇数长度，补充一个值
            x = F.pad(x, (0, 1), mode='reflect')
        
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class AdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)

        # Upsample c and d to match the input size
        c = F.interpolate(c, size=x.size(2), mode='linear', align_corners=True)
        d = F.interpolate(d, size=x.size(2), mode='linear', align_corners=True)

        r = None
        if (self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(c)  # Use the upsampled c as the output
        d = self.norm_d(d)

        return x, r, d # x：信号的低频部分，表示经过小波变换后的近似部分。r：正则化项，衡量信号中低频部分和高频部分之间的差异。 d：信号的高频部分，表示细节部分或噪声。


class Config:
    def __init__(self):
        self.enc_in = 64  # 输入通道数
        self.lifting_kernel_size = 4  # 卷积核大小
        self.regu_details = 0.1  # 细节部分的正则化强度
        self.regu_approx = 0.1  # 近似部分的正则化强度


class DWCA(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.heads = heads
        self.cross_attn = self._create_attention_module(d_model, heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            wn(nn.Linear(d_model, d_model * 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            wn(nn.Linear(d_model * 4, d_model))
        )
        
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            wn(nn.Linear(d_model * 2, d_model)),
            nn.GELU()
        )
        
        self.res_scale = nn.Parameter(torch.ones(1))

        d_keys = d_model // heads
        self.query_projection = wn(nn.Linear(d_model, d_keys * heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * heads))
        self.scale = d_keys ** -0.5
        
        self.temporal_mixer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.adaptive_weights = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.LayerNorm(2),
            nn.Softmax(dim=-1)
        )
        
        self.weight_scaling = 0.1

    def _create_attention_module(self, d_model, heads, dropout):

        d_model_adjusted = (d_model // heads) * heads
        d_k = d_model_adjusted // heads
        
        # 创建一个完整的注意力模块
        class AttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 基本参数
                self.d_model = d_model_adjusted
                self.d_k = d_k
                self.heads = heads
                self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([d_k])))
                
                # 小波变换
                config = Config()
                config.enc_in = d_model
                config.lifting_kernel_size = 4
                config.regu_details = 0.1
                config.regu_approx = 0.1
                self.wavelet = AdpWaveletBlock(config, d_model)
                
                # 投影层
                self.q_proj = wn(nn.Linear(d_model, d_model_adjusted, bias=False))
                self.k_proj = wn(nn.Linear(d_model, d_model_adjusted, bias=False))
                self.v_proj = wn(nn.Linear(d_model, d_model_adjusted, bias=False))
                self.out_proj = wn(nn.Linear(d_model_adjusted, d_model, bias=False))
                
                # 归一化层 - 确保使用正确的d_model
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
                # 特征融合
                self.fusion = SqueezeAndExciteFusionAdd1D(d_model)
            
            def forward(self, query, key, value, mask=None):
                batch_size = query.shape[0]
                seq_len_q = query.shape[1]
                seq_len_k = key.shape[1]
                
                # 确保scale在正确的设备上
                if self.scale.device != query.device:
                    self.scale = self.scale.to(query.device)
                
                # 层归一化
                query = self.norm1(query)
                key = self.norm1(key)
                value = self.norm1(value)
                
                # 应用小波变换
                B, L, D = query.shape
                query_wave = query.transpose(1, 2)  # [B, D, L]
                key_wave = key.transpose(1, 2)
                
                # 确保长度匹配
                if query_wave.size(2) != key_wave.size(2):
                    max_len = max(query_wave.size(2), key_wave.size(2))
                    if query_wave.size(2) < max_len:
                        query_wave = F.pad(query_wave, (0, max_len - query_wave.size(2)))
                    if key_wave.size(2) < max_len:
                        key_wave = F.pad(key_wave, (0, max_len - key_wave.size(2)))
                
                # 小波变换
                query_wave, _, _ = self.wavelet(query_wave)
                key_wave, _, _ = self.wavelet(key_wave)
                
                # 转回原始维度并截断到正确长度
                query_wave = query_wave.transpose(1, 2)[:, :L, :]
                key_wave = key_wave.transpose(1, 2)[:, :key.size(1), :]
                
                # 特征融合
                query = self.fusion(query, query_wave)
                key = self.fusion(key, key_wave)
                
                # 投影
                Q = self.q_proj(query)
                K = self.k_proj(key)
                V = self.v_proj(value)
                
                # 多头注意力
                Q = Q.view(batch_size, seq_len_q, self.heads, self.d_k).transpose(1, 2)
                K = K.view(batch_size, seq_len_k, self.heads, self.d_k).transpose(1, 2)
                V = V.view(batch_size, seq_len_k, self.heads, self.d_k).transpose(1, 2)
                
                # 计算注意力
                scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                attn = self.dropout(F.softmax(scores, dim=-1))
                
                # 应用注意力
                out = torch.matmul(attn, V)
                out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
                
                # 输出投影和归一化
                out = self.out_proj(out)
                out = self.norm2(out)
                
                return out, attn
        
        return AttentionModule()
    
    def forward(self, basis, series):
        B, N, D = basis.shape
        _, C, _ = series.shape
        
        # 特征提取
        series_features = torch.cat([
            series.mean(dim=1),
            series.max(dim=1)[0]
        ], dim=-1)  # [B, D*2]
        
        # 时序融合
        temporal_context = self.temporal_mixer(series_features).unsqueeze(1)  # [B, 1, D]
        
        # 计算权重
        weights = self.adaptive_weights(temporal_context) * self.weight_scaling
        
        # 增强输入
        enhanced_basis = basis + weights[:, :, 0:1] * temporal_context
        enhanced_series = series + weights[:, :, 1:2] * temporal_context.repeat(1, C, 1)
        
        # 双向注意力
        basis_attn, attn1 = self.cross_attn(enhanced_basis, enhanced_series, enhanced_series)
        series_attn, attn2 = self.cross_attn(enhanced_series, enhanced_basis, enhanced_basis)
        
        # 特征融合
        basis_fused = self.feature_fusion(torch.cat([enhanced_basis, basis_attn], dim=-1))
        series_fused = self.feature_fusion(torch.cat([enhanced_series, series_attn], dim=-1))
        
        # 残差连接
        basis = enhanced_basis + self.res_scale * self.ffn(basis_fused)
        series = enhanced_series + self.res_scale * self.ffn(series_fused)
        
        # 最终系数计算
        B, L, _ = series.shape
        B, S, _ = basis.shape
        H = self.heads
        
        queries = self.query_projection(series).view(B, L, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(basis).view(B, S, H, -1).permute(0, 2, 1, 3)
        
        coef = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        
        return coef, [attn1], [attn2]

