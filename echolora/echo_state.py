import torch


class EchoState:
    def __init__(self):
        super().__init__()
        self.storage_dict = {}

    def store(self, layer_idx, hidden_state):
        self.storage_dict[layer_idx] = hidden_state

    def get_echo(self):
        states = torch.stack(list(self.storage_dict.values()))
        return states.mean(dim=0)

    def clear(self):
        self.storage_dict = {}
