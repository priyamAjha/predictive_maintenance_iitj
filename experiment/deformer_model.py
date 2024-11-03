import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        #Input shape to MultiHeadSelfAttention: torch.Size([32, 20, 64])
        # QKV shape after linear layer: torch.Size([32, 20, 192])
    def forward(self, x):
        N, T, E = x.shape
        # print(f"Input shape to MultiHeadSelfAttention: {x.shape}")
        qkv = self.qkv(x)  # Shape: (N, T, 3 * embed_dim)
        # print(f"QKV shape after linear layer: {qkv.shape}")
        #embed_dim=64, num_heads=4, head_dim = 64//4
        qkv = qkv.reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) #[3, 32, 4, 20, 16]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = torch.einsum("nhqd, nhkd -> nhqk", [q, k]) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        out = torch.einsum("nhqk, nhvd -> nhqd", [attn_weights, v]).reshape(N, T, E)
        # print(f"Output shape from multiheadself_attn: {out.shape}")
        return self.fc_out(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion=4, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # print(f"trans_encoder_attn_input: {x.shape}")
        attn_out = self.attention(x)
        # print(f"trans_encoder_attn_out : {attn_out.shape}")
        x = self.norm1(x + self.dropout(attn_out))
        # print(f"trans_encoder_dr_residual_norm1: {x.shape}")
        ff_out = self.feed_forward(x)
        # print(f"trans_encoder_feed_forward: {ff_out.shape}")
        x = self.norm2(x + self.dropout(ff_out))
        # print(f"trans_encoder_last_layer: {x.shape}")
        return x

class DEformer(nn.Module):
    # model = DEformer(input_dim=16, embed_dim=64, num_heads=4, num_layers=2, output_dim=1, seq_len=20)
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, seq_len):
        super(DEformer, self).__init__()
        
        # Adjusted projection layers to match dimensions
        self.temporal_projection = nn.Linear(input_dim, embed_dim)  # Embedding each feature's sequence
        self.spatial_projection = nn.Linear(seq_len, embed_dim)  # Embedding across features
        
        self.temporal_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.spatial_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(embed_dim, output_dim)  # Double embed_dim due to concatenation
        
    def forward(self, x):
        # Temporal encoding: Apply temporal projection across the feature dimension
        # print(f"Input_shape: {x.shape}")
        x_temporal = self.temporal_projection(x)  # Shape: (batch, seq_len, embed_dim)
        # print(f"Shape after temporal projection: {x_temporal.shape}")
        for layer in self.temporal_encoders:
            x_temporal = layer(x_temporal)

        # Spatial encoding: Apply spatial projection across the sequence dimension
        x_spatial = self.spatial_projection(x.transpose(1, 2)) #.transpose(1, 2)  # Shape: (batch, seq_len, embed_dim)
        # print(f"Shape after spatial projection: {x_spatial.shape}")
        for layer in self.spatial_encoders:
            x_spatial = layer(x_spatial)

        # Concatenate both encodings along the last dimension
        # combined = torch.cat((x_temporal, x_spatial), dim=-1)  # Shape: (batch, seq_len, embed_dim * 2)


        # Summing both encodings
        combined = x_temporal + x_spatial  # Shape: (batch, seq_len, embed_dim)
        
        # Apply the final output projection
        # print(f"combined.mean_shape: {combined.mean(dim=1).shape}, combined_shape: {combined.shape}")
        out = self.output_projection(combined.mean(dim=1))  # Shape: (batch, output_dim)
        
        return out
    
if __name__ == '__main__':
    # Hyperparameters
    input_dim = 16        # Number of features
    seq_len = 16 #20          # Sequence length
    embed_dim = 64        # Embedding dimension
    num_heads = 4         # Number of attention heads
    num_layers = 2        # Number of Transformer layers in each encoder
    output_dim = 1        # Output dimension for forecasting

    # Instantiate and test model
    # model = DEformer(input_dim=16, embed_dim=64, num_heads=4, num_layers=2, output_dim=1, seq_len=20)
    model = DEformer(input_dim, embed_dim, num_heads, num_layers, output_dim, seq_len)
    x = torch.randn(32, seq_len, input_dim)  # (batch, time, features)
    out = model(x)
    # print("Output shape:", out.shape)  # Expected output: (batch, output_dim)
