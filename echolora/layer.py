import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


class EchoLoraLinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha, lora_dropout=0.0, bottleneck_dim=64):
        super().__init__()

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if isinstance(base_layer, Conv1D):
            # Conv1D stores (in_features, out_features)
            in_features, out_features = base_layer.weight.shape
        else:
            # nn.Linear stores (out_features, in_features)
            out_features, in_features = base_layer.weight.shape

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.scaling = lora_alpha / r

        # echo projection network
        self.echo_proj1 = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.echo_proj2 = nn.Linear(bottleneck_dim, out_features, bias=False)

        # echo gate network
        self.echo_gate1 = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.echo_gate2 = nn.Linear(bottleneck_dim, out_features, bias=True)

        # learnable scale
        self.lambda_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Reset parameters
        self.reset_parameters()

        self.echo_signal = None
        self.echo_mask = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        # Echo projection - Zeros so echo starts neutral
        nn.init.zeros_(self.echo_proj1.weight)
        nn.init.zeros_(self.echo_proj2.weight)
        # Echo gate - Zeros for weights, -2.0 for bias
        nn.init.zeros_(self.echo_gate1.weight)
        nn.init.zeros_(self.echo_gate2.weight)
        nn.init.constant_(self.echo_gate2.bias, -2.0)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling

        result = base_output + lora_output

        if self.echo_signal is not None:
            # normalize
            z = torch.nn.functional.layer_norm(
                self.echo_signal, self.echo_signal.shape[-1:]
            )
            z = z.unsqueeze(1)
            e = self.echo_proj2(torch.tanh(self.echo_proj1(z)))
            g = torch.sigmoid(self.echo_gate2(torch.tanh(self.echo_gate1(z))))
            delta = self.lambda_scale * (e * g)
            mask = self.echo_mask.unsqueeze(-1)
            result = result + mask * delta

        return result
