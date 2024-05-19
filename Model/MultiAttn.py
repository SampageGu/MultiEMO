import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Bidirectional cross-attention layers.
'''
class BidirectionalCrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)
    

    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim = -1), V)

        return attention
    

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)

        return attention


'''
Multi-head bidirectional cross-attention layers.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)
    

    def forward(self, query, key, value):
        heads = [self.attention_heads[i](query, key, value) for i in range(self.num_heads)]
        multihead_attention = self.projection_matrix(torch.cat(heads, dim = -1))

        return multihead_attention


'''
A feed-forward network, which operates as a key-value memory.
'''
class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))


'''
Residual connection to smooth the learning process.
'''
class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))

        return output


class MultiAttnLayer(nn.Module):
    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.attn_1 = MultiHeadAttention(num_heads, model_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)

    def forward(self, query_modality, modality_A, modality_B):
        # First attention mechanism with add & norm
        attn_output_1 = self.add_norm_1(query_modality, lambda x: self.attn_1(x, modality_A, modality_A))
        
        # Second attention mechanism with add & norm
        attn_output_2 = self.add_norm_2(attn_output_1, lambda x: self.attn_2(x, modality_B, modality_B))
        
        # Feedforward with add & norm
        ff_output = self.add_norm_3(attn_output_2, self.ff)
        
        return ff_output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output
    
    
class Feedforward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
'''
Stacks of MultiAttn layers.
'''
class MultiAttn(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.multiattn_layers = nn.ModuleList([
            MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)
        ])

    def forward(self, query_modality, modality_A, modality_B):
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(query_modality, modality_A, modality_B)
        return query_modality


class MultiAttnModel(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.multiattn_audio_visual = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_text = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_audio = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_visual = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
    
    def forward(self, text_features, audio_features, visual_features):
        # Step 1: Fuse audio and visual modalities
        fused_audio_visual = self.multiattn_audio_visual(audio_features, visual_features, visual_features)
        
        # Step 2: Fuse the audio-visual with textual modality
        fused_text = self.multiattn_text(text_features, fused_audio_visual, fused_audio_visual)
        fused_audio = self.multiattn_audio(fused_audio_visual, text_features, fused_audio_visual)
        fused_visual = self.multiattn_visual(fused_audio_visual, visual_features, fused_audio_visual)

        return fused_text, fused_audio, fused_visual
