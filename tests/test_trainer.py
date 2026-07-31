import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from echolora.trainer import EchoLoraTrainer
from echolora.config import EchoLoraConfig


class DummyOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


def build_model():
    config = GPT2Config(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    return GPT2LMHeadModel(config)


def build_batch():
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()
    return input_ids, labels


def test_model_forward_has_grad():
    model = build_model()

    input_ids, labels = build_batch()

    out = model(input_ids=input_ids, labels=labels)

    assert out.loss.requires_grad
    assert out.logits.requires_grad


def test_gradients_enabled():
    assert torch.is_grad_enabled()


def test_loss_backward():
    model = build_model()

    input_ids, labels = build_batch()

    out = model(input_ids=input_ids, labels=labels)

    out.loss.backward()

    grad_found = False

    for p in model.parameters():
        if p.grad is not None:
            grad_found = True
            break

    assert grad_found



if __name__ == "__main__":
    print("Test 1: Gradients globally enabled")
    test_gradients_enabled()
    print("PASS")

    print("\nTest 2: Model forward produces differentiable loss")
    test_model_forward_has_grad()
    print("PASS")

    print("\nTest 3: Backward computes gradients")
    test_loss_backward()
    print("PASS")

    print("\nAll trainer tests passed.")


    print("\nTest 5: EchoLoraTrainer train_steps")

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    config = EchoLoraConfig(
        source_layers=[0],
        target_layers=[1],
    )

    trainer = EchoLoraTrainer(
        model=model,
        config=config,
        optimizer=optimizer,
    )

    input_ids, labels = build_batch()

    loss = trainer.train_steps(
        input_ids=input_ids,
        labels=labels,
        total_steps=10,
    )

    print("Returned loss:", loss)
    assert isinstance(loss, float)

    print("PASS")