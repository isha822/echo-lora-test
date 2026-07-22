import torch
import torch.nn as nn

class EchoLoraLinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha, lora_dropout=0.0):
        super().__init__()
        
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        out_features, in_features = self.base_layer.weight.shape

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.scaling = lora_alpha / r 
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output