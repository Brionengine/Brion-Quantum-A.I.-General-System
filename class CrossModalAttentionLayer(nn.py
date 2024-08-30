from torch import nn

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CrossModalAttentionLayer, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultithreadAttention(hidden_dim, num_heads=8)

    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        combined_features, _ = self.attention(text_proj, image_proj, image_proj)
        return combined_features
