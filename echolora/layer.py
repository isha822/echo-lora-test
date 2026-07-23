import torch
import torch.nn as nn

class EchoLoraLinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha, lora_dropout=0.0, bottleneck_dim=64):
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
        
        #echo projection network
        self.echo_proj1 = nn.Linear(in_features, bottleneck_dim, bias = False)
        self.echo_proj2 = nn.Linear(bottleneck_dim, in_features, bias = False)
        
        #echo gate network
        self.echo_gate1 = nn.Linear(in_features, bottleneck_dim, bias = False)
        self.echo_gate2 = nn.Linear(bottleneck_dim, in_features, bias = True)
        
        #learnable scale
        self.lambda_scale = nn.Parameter(torch.ones(1) * 0.1)

        #Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        #Echo projection - Zeros so echo starts neutral
        nn.init.zeros_(self.echo_proj1.weight)
        nn.init.zeros_(self.echo_proj2.weight)
        #Echo gate - Zeros for weights, -2.0 for bias
        nn.init.zeros_(self.echo_gate1.weight)
        nn.init.zeros_(self.echo_gate2.weight)
        nn.init.constant_(self.echo_gate2.bias, -2.0)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output