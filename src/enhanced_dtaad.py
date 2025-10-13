import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGCardiacAttention(nn.Module):
    """ECG-specific attention mechanism focusing on cardiac patterns - Single Feature Compatible"""
    def __init__(self, input_dim, hidden_dim=64):
        super(ECGCardiacAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # For single feature ECG, we need to expand the feature dimension first
        if input_dim == 1:
            self.feature_expander = nn.Linear(input_dim, hidden_dim)
            attention_dim = hidden_dim
        else:
            self.feature_expander = None
            attention_dim = input_dim
        
        # Ensure attention_dim is divisible by num_heads
        num_heads = min(4, attention_dim)  # Adaptive number of heads
        if attention_dim % num_heads != 0:
            # Adjust to nearest divisible number
            attention_dim = (attention_dim // num_heads) * num_heads
            if attention_dim == 0:
                attention_dim = num_heads
        
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Project to attention dimension if needed
        if input_dim == 1 or attention_dim != (hidden_dim if input_dim == 1 else input_dim):
            self.input_projection = nn.Linear(hidden_dim if input_dim == 1 else input_dim, attention_dim)
        else:
            self.input_projection = None
        
        # Cardiac rhythm-specific attention
        self.rhythm_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Heartbeat pattern attention  
        self.heartbeat_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=max(1, num_heads // 2), 
            batch_first=True
        )
        
        self.fusion_layer = nn.Linear(attention_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch, features, sequence]
        batch_size, features, seq_len = x.shape
        
        # Expand features if single feature ECG
        if self.feature_expander is not None:
            x_perm = x.permute(0, 2, 1)  # [batch, sequence, features]
            x_expanded = self.feature_expander(x_perm)  # [batch, sequence, hidden_dim]
        else:
            x_expanded = x.permute(0, 2, 1)  # [batch, sequence, features]
        
        # Project to attention dimension if needed
        if self.input_projection is not None:
            x_projected = self.input_projection(x_expanded)
        else:
            x_projected = x_expanded
        
        # Rhythm-level attention (longer patterns)
        rhythm_out, _ = self.rhythm_attention(x_projected, x_projected, x_projected)
        
        # Heartbeat-level attention (shorter patterns)  
        heartbeat_out, _ = self.heartbeat_attention(x_projected, x_projected, x_projected)
        
        # Fuse both attention outputs
        fused = torch.cat([rhythm_out, heartbeat_out], dim=-1)
        fused = self.fusion_layer(fused)
        fused = self.dropout(fused)
        
        return fused.permute(0, 2, 1)  # Back to [batch, features, sequence]

class MultiScaleTemporalExtractor(nn.Module):
    """Multi-scale temporal feature extraction for ECG signals - Single Feature Compatible"""
    def __init__(self, input_dim, scales=[3, 5, 7]):
        super(MultiScaleTemporalExtractor, self).__init__()
        self.scales = scales
        
        # For single feature, use regular convolutions instead of grouped
        if input_dim == 1:
            self.extractors = nn.ModuleList([
                nn.Conv1d(input_dim, input_dim, kernel_size=scale, padding=scale//2)
                for scale in scales
            ])
        else:
            self.extractors = nn.ModuleList([
                nn.Conv1d(input_dim, input_dim, kernel_size=scale, 
                         padding=scale//2, groups=input_dim)
                for scale in scales
            ])
        
        # Fusion layer
        self.fusion = nn.Conv1d(input_dim * len(scales), input_dim, 1)
        self.norm = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        # x shape: [batch, features, sequence]
        multi_scale_features = []
        
        for extractor in self.extractors:
            features = extractor(x)
            multi_scale_features.append(features)
        
        # Concatenate all scales
        concatenated = torch.cat(multi_scale_features, dim=1)
        
        # Fuse to original dimension
        fused = self.fusion(concatenated)
        fused = self.norm(fused)
        
        return fused

class EnhancedDTAAD(nn.Module):
    """Enhanced DTAAD with ECG-specific improvements - Single Feature Compatible"""
    def __init__(self, feats):
        super(EnhancedDTAAD, self).__init__()
        
        # Import and create base DTAAD model as a component
        from src.models import DTAAD
        self.base_dtaad = DTAAD(feats)
        
        # Copy essential attributes from base model
        self.n_window = getattr(self.base_dtaad, 'n_window', 10)
        self.lr = getattr(self.base_dtaad, 'lr', 1e-4)
        self.batch = getattr(self.base_dtaad, 'batch', 64)
        
        # ECG-specific components (now compatible with single feature)
        self.ecg_attention = ECGCardiacAttention(feats)
        self.multi_scale_extractor = MultiScaleTemporalExtractor(feats)
        
        # Enhanced attention mechanism (adaptive for single feature)
        if feats == 1:
            # For single feature, expand to 8 dimensions for multi-head attention
            self.feature_expander = nn.Linear(1, 8)
            attention_embed_dim = 8
            attention_heads = 8
        else:
            self.feature_expander = None
            attention_embed_dim = feats
            attention_heads = min(8, feats) if feats % min(8, feats) == 0 else 1
        
        self.enhanced_attention = nn.MultiheadAttention(
            embed_dim=attention_embed_dim, 
            num_heads=attention_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Project back to original dimension if expanded
        if feats == 1:
            self.feature_projector = nn.Linear(8, 1)
        else:
            self.feature_projector = None
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feats, max(1, feats // 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(1, feats // 2), 1),
            nn.Sigmoid()
        )
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.name = 'Enhanced_DTAAD'
        
    def forward(self, x):
        # Get base DTAAD outputs
        tcn_out1, tcn_out2 = self.base_dtaad(x)
        
        # ECG-specific enhancements
        cardiac_features = self.ecg_attention(x)
        multi_scale_features = self.multi_scale_extractor(x)
        
        # Enhanced attention on original outputs
        if self.feature_expander is not None:
            # Expand single feature for multi-head attention
            tcn_perm = tcn_out2.permute(0, 2, 1)  # [batch, seq, features]
            tcn_expanded = self.feature_expander(tcn_perm)  # [batch, seq, 8]
            enhanced_out, attention_weights = self.enhanced_attention(
                tcn_expanded, tcn_expanded, tcn_expanded
            )
            enhanced_out = self.feature_projector(enhanced_out)  # Back to [batch, seq, 1]
            enhanced_out = enhanced_out.permute(0, 2, 1)  # [batch, 1, seq]
        else:
            tcn_perm = tcn_out2.permute(0, 2, 1)  # [batch, seq, features]
            enhanced_out, attention_weights = self.enhanced_attention(
                tcn_perm, tcn_perm, tcn_perm
            )
            enhanced_out = enhanced_out.permute(0, 2, 1)  # Back to [batch, features, seq]
        
        # Adaptive fusion of all paths
        w1, w2, w3 = F.softmax(self.fusion_weights, dim=0)
        
        # Combine all enhancement paths with size matching
        min_size1 = min(tcn_out1.size(2), cardiac_features.size(2))
        min_size2 = min(tcn_out2.size(2), multi_scale_features.size(2), enhanced_out.size(2))
        
        cardiac_resized = cardiac_features[:, :, :min_size1]
        multi_scale_resized = multi_scale_features[:, :, :min_size2]
        enhanced_resized = enhanced_out[:, :, :min_size2]
        
        enhanced_tcn1 = tcn_out1[:, :, :min_size1] + w1 * cardiac_resized
        enhanced_tcn2 = tcn_out2[:, :, :min_size2] + w2 * multi_scale_resized + w3 * enhanced_resized
        
        return enhanced_tcn1, enhanced_tcn2
    
    def get_attention_weights(self, x):
        """Extract attention weights for visualization"""
        cardiac_features = self.ecg_attention(x)
        multi_scale_features = self.multi_scale_extractor(x)
        
        # Get enhanced attention weights
        tcn_out1, tcn_out2 = self.base_dtaad(x)
        
        if self.feature_expander is not None:
            tcn_perm = tcn_out2.permute(0, 2, 1)
            tcn_expanded = self.feature_expander(tcn_perm)
            _, attention_weights = self.enhanced_attention(tcn_expanded, tcn_expanded, tcn_expanded)
        else:
            tcn_perm = tcn_out2.permute(0, 2, 1)
            _, attention_weights = self.enhanced_attention(tcn_perm, tcn_perm, tcn_perm)
        
        return {
            'enhanced_attention': attention_weights,
            'fusion_weights': F.softmax(self.fusion_weights, dim=0),
            'cardiac_features': cardiac_features,
            'multi_scale_features': multi_scale_features
        }

# For compatibility with your existing framework
class Enhanced_DTAAD(EnhancedDTAAD):
    """Wrapper class for compatibility"""
    pass