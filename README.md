# EchoLoRA

A PyTorch implementation of **EchoLoRA**, a parameter-efficient fine-tuning (PEFT) method that enhances LoRA by injecting contextual representations from earlier transformer layers into later layers during training.

> This repository is an independent implementation based on the EchoLoRA paper and is intended for research, experimentation, and reproducibility.

---

## Overview

EchoLoRA extends LoRA by introducing an **Echo Path** that captures intermediate hidden representations from source transformer layers and injects processed contextual information into target layers.

The implementation includes:

- Standard LoRA adapters
- Echo projection network
- Echo gating mechanism
- Stochastic routing
- Teacher–student knowledge distillation
- Boundary-token based echo extraction
- End-to-end training pipeline for causal language models

---

## Features

- Parameter-efficient fine-tuning using LoRA
- Echo signal extraction using forward hooks
- Projection and gating networks
- Adaptive routing probability
- Knowledge distillation loss
- GPT-2 compatible implementation
- Modular design compatible with Hugging Face Transformers
- Integration example and unit tests

---

## Repository Structure

```text
echolora/
│
├── config.py          # EchoLoRA configuration
├── layer.py           # EchoLoRA linear layer
├── model.py           # Inject EchoLoRA into transformer
├── trainer.py         # Two-pass training pipeline
├── echo_state.py      # Hidden-state storage
├── utils.py           # Boundary extraction utilities
│
tests/
examples/
```

---

## EchoLoRA Pipeline

```
Input
   │
   ▼
Forward Pass (Echo OFF)
   │
   ├── Store hidden states from source layers
   ▼
Boundary hidden representation
   │
   ▼
Projection Network
   │
Gate Network
   │
Echo Injection
   │
   ▼
Forward Pass (Echo ON)
   │
Cross Entropy Loss
+
Knowledge Distillation Loss
   │
   ▼
Backpropagation
```

---

## Installation

```bash
git clone <repo-url>
cd echo-lora

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Integration Check

Run a complete end-to-end sanity check:

```bash
PYTHONPATH=. python3 examples/integration_check.py
```

The script verifies:

- GPT-2 loads successfully
- EchoLoRA modules are injected
- One training step executes
- Gradients propagate correctly
- Echo state is cleaned after training

---

## Testing

Run the unit tests:

```bash
PYTHONPATH=. python3 tests/test_layer.py
PYTHONPATH=. python3 tests/test_trainer.py
```

---

## Current Status

- ✅ EchoLoRA layer implementation
- ✅ Configuration system
- ✅ Echo projection and gating
- ✅ Hidden-state extraction
- ✅ Two-pass training pipeline
- ✅ Knowledge distillation
- ✅ GPT-2 integration
- ✅ Integration testing

---

## References

EchoLoRA: *Parameter-Efficient Fine-Tuning via Echo Representations* (2026)

---

## License

This project is intended for research and educational purposes.
