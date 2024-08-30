class HybridModel(nn.Module):
    def __init__(self, vit_model, clip_model):
        super(HybridModel, self).__init__()
        self.vit = vit_model
        self.clip = clip_model
        self.fusion_layer = nn.Linear(vit_model.config.hidden_size + clip_model.config.hidden_size, 512)

    def forward(self, image, text):
        vit_output = self.vit(image)
        clip_output = self.clip(text)
        combined_output = torch.cat((vit_output, clip_output), dim=-1)
        return self.fusion_layer(combined_output)
