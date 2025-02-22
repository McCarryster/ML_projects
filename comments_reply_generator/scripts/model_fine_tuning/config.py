train_args = {
    'num_epochs': 1,
    'batch_size': 2,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 1,
    'gradient_clip': 1.0,
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'device': 'cpu',
    'mixed_precision': False,
    'save_steps': 5,
    'logging_steps': 1,
    'validation_steps': 5,
    'early_stopping': False,
    'early_stopping_patience': 3
}

evaluation_args = {
    'max_length': 1000,
    'max_new_tokens': 128,
    'num_beams': 3,
    'top_k': 5, # 50
    'temperature': 0.6,
    'top_p': 0.9,
    'no_repeat_ngram_size': 3
}