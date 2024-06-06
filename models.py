import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute positional encodings
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.positional_encoding = torch.zeros(max_seq_len, d_model)
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
    
    def forward(self, x):
        # Add positional encodings to input embeddings
        x = x + self.positional_encoding[:, :x.size(1)].to(x.device)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        
        query = self.split_heads(self.query(query), batch_size)
        key = self.split_heads(self.key(key), batch_size)
        value = self.split_heads(self.value(value), batch_size)
        
        scaled_attention_logits = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scaled_attention_logits += mask * -1e9
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc(output)

class NormLayer(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super(NormLayer, self).__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = NormLayer(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = NormLayer(d_model)
    
    def forward(self, x, mask):
        attention_output = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.norm1(x + attention_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerTranslator(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, hidden_dim, input_vocab_size, target_vocab_size, max_seq_len, dropout=0.1, learning_rate=1e-3, batch_size=128):
        super(TransformerTranslator, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(d_model, target_vocab_size)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = self.positional_encoder(x)
        x = self.encoder(x, mask)
        x = self.fc(x)
        return x

