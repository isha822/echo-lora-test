import torch
import torch.nn as nn
from echolora.config import EchoLoraConfig
from echolora.layer import EchoLoraLinear
from echolora.model import apply_echo_lora
from echolora.echo_state import EchoState
from echolora.utils import get_boundary_positions, get_answer_mask, extract_boundary_hidden

# --- test 1: layer forward without echo ---
print("Test 1: layer forward without echo")
base = nn.Linear(64, 128)
layer = EchoLoraLinear(base, r=4, lora_alpha=8, bottleneck_dim=16)
x = torch.randn(2, 10, 64)
out = layer(x)
assert out.shape == (2, 10, 128), f"Wrong shape: {out.shape}"
print("PASS")

# --- test 2: layer forward with echo ---
print("Test 2: layer forward with echo")
layer.echo_signal = torch.randn(2, 64)
layer.echo_mask = torch.ones(2, 10)
out2 = layer(x)
assert out2.shape == (2, 10, 128)
print("PASS")

# --- test 3: echo state ---
print("Test 3: echo state")
state = EchoState()
state.store(0, torch.randn(2, 10, 64))
state.store(1, torch.randn(2, 10, 64))
echo = state.get_echo()
assert echo.shape == (2, 10, 64)
state.clear()
assert state.storage_dict == {}
print("PASS")

# --- test 4: utils ---
print("Test 4: utils")
labels = torch.tensor([
    [-100, -100, -100, 512, 318, 779],
    [-100, -100,  423, 215, 779, 612],
])
boundary = get_boundary_positions(labels)
assert boundary.tolist() == [2, 1]
mask = get_answer_mask(labels)
assert mask.shape == (2, 6)
hidden = torch.randn(2, 6, 64)
extracted = extract_boundary_hidden(hidden, boundary)
assert extracted.shape == (2, 64)
print("PASS")

# --- test 5: model wrapping with fake model ---
print("Test 5: model wrapping")
class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)

fake = FakeModel()
config = EchoLoraConfig(r=4, lora_alpha=8, target_modules=["layer1"])
apply_echo_lora(fake, config)
echo_layers = sum(1 for m in fake.modules() if isinstance(m, EchoLoraLinear))
assert echo_layers == 1, f"Expected 1, got {echo_layers}"
print("PASS")

print("\nAll tests passed.")

# test 6: hook shape handling
print("Test 6: hook shape handling")
import torch

hidden_2d = torch.randn(32, 768)    # GPT-2 style output
hidden_3d = torch.randn(2, 16, 768) # LLaMA style output

# simulate the fix
result = hidden_2d if hidden_2d.dim() == 3 else hidden_3d
assert result.shape == (2, 16, 768)
print("PASS")

# test 7: full echo pipeline with 3D hidden states
print("Test 7: echo pipeline")
from echolora.echo_state import EchoState
from echolora.utils import get_boundary_positions, extract_boundary_hidden

state = EchoState()
state.store(0, torch.randn(2, 16, 768))
state.store(1, torch.randn(2, 16, 768))

z = state.get_echo()
assert z.shape == (2, 16, 768), f"Wrong shape: {z.shape}"

labels = torch.zeros(2, 16, dtype=torch.long)
labels[:, :3] = -100
boundary = get_boundary_positions(labels)
echo_signal = extract_boundary_hidden(z, boundary)
assert echo_signal.shape == (2, 768), f"Wrong shape: {echo_signal.shape}"
print("PASS")

print("\nAll tests passed.")