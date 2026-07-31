import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import EchoLoraLinear
from .echo_state import EchoState
from .utils import get_boundary_positions, get_answer_mask, extract_boundary_hidden


class EchoLoraTrainer:
    def __init__(self, model, config, optimizer):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.echo_state = EchoState()
        self.current_step = 0

    def _get_routing_prob(self, total_steps):
        k = self.current_step
        p_start = self.config.p_start
        p_end = self.config.p_end
        return p_start + (k / (total_steps - 1)) * (p_end - p_start)

    def _register_source_hooks(self):
        handles = []
        for layer_idx in self.config.source_layers:
            layer = self.model.transformer.h[layer_idx]

            def hook(module, input, output, idx=layer_idx):
                # output[0] shape varies by architecture
                # if 2D (batch*seq, hidden) → use input[0] instead
                # if 3D (batch, seq, hidden) → use directly
                hidden = output[0] if output[0].dim() == 3 else input[0]
                self.echo_state.store(idx, hidden.detach())

            handles.append(layer.register_forward_hook(hook))
        return handles

    def _set_echo_on_target_layers(self, echo_signal, answer_mask):
        for module in self.model.modules():
            if isinstance(module, EchoLoraLinear):
                module.echo_signal = echo_signal
                module.echo_mask = answer_mask

    def _clear_echo_on_target_layers(self):
        for module in self.model.modules():
            if isinstance(module, EchoLoraLinear):
                module.echo_signal = None
                module.echo_mask = None

    def _compute_kd_loss(self, logits_off, logits_on, answer_mask):
        tau = self.config.tau
        q_off = F.softmax(logits_off / tau, dim=-1)
        q_on = F.softmax(logits_on / tau, dim=-1)
        kl = F.kl_div(q_off.log(), q_on, reduction="none").sum(-1)
        kl = (kl * answer_mask).sum() / answer_mask.sum()
        return self.config.lambda_kd * (tau**2) * kl

    def train_steps(self, input_ids, labels, total_steps):
        self.optimizer.zero_grad()

        # pass 1: collect hidden states (no gradients)
        handles = self._register_source_hooks()
        with torch.no_grad():
            self.model(input_ids=input_ids, labels=labels)
        for h in handles:
            h.remove()

        # get echo signal
        z = self.echo_state.get_echo()
        boundary = get_boundary_positions(labels)
        echo_signal = extract_boundary_hidden(z, boundary)
        answer_mask = get_answer_mask(labels)

        # always run echo-off pass WITH gradients for loss_off
        out_off = self.model(input_ids=input_ids, labels=labels)
        loss_off = out_off.loss
        logits_off = out_off.logits.detach()

        # stochastic routing
        p_k = self._get_routing_prob(total_steps)
        rk = torch.bernoulli(torch.tensor(p_k)).item()

        total_loss = loss_off

        if rk == 1:
            self._set_echo_on_target_layers(echo_signal, answer_mask)
            out_on = self.model(input_ids=input_ids, labels=labels)
            loss_on = out_on.loss
            logits_off = out_off.logits.detach()
            logits_on = out_on.logits
            kd_loss = self._compute_kd_loss(logits_off, logits_on, answer_mask)
            total_loss = loss_off + loss_on + self.config.lambda_kd * kd_loss

        total_loss.backward()
        self.optimizer.step()
        self._clear_echo_on_target_layers()
        self.echo_state.clear()
        self.current_step += 1

        return total_loss.item()
