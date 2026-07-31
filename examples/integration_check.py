def main():
    import torch
    from transformers import GPT2LMHeadModel
    from echolora.config import EchoLoraConfig
    from echolora.model import apply_echo_lora
    from echolora.trainer import EchoLoraTrainer
    from echolora.layer import EchoLoraLinear

    print('1. Loading model...')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print('✓ 1. Model loaded')

    config = EchoLoraConfig(target_modules=['c_attn'], p_start=1.0, p_end=1.0)
    print('✓ 2. Config created')

    model = apply_echo_lora(model, config)
    print('✓ 3. EchoLoRA applied')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    print('✓ 4. Optimizer created')

    trainer = EchoLoraTrainer(model, config, optimizer)
    print('✓ 5. Trainer created')

    input_ids = torch.randint(0, 50257, (2, 32))
    labels = input_ids.clone()
    labels[:, :10] = -100
    print('✓ 6. Batch created')

    print('7a. Calling train_steps()')
    loss = trainer.train_steps(input_ids, labels, total_steps=100)
    print('7b. train_steps() returned')

    print('Loss:', loss)

    echo_layers = [m for m in model.modules() if isinstance(m, EchoLoraLinear)]
    print('EchoLoRA layers:', len(echo_layers))

    assert len(echo_layers) > 0
    print('✓ EchoLoRA layers found')

    print('Echo signal cleared:', echo_layers[0].echo_signal is None)
    assert echo_layers[0].echo_signal is None

    print('✓ Echo signal cleared')
    print('✓ Integration test PASSED')

if __name__ =="__main__":
    main()