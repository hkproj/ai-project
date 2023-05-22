from pathlib import Path

def get_config():
    return {
        'batch_size': 16,
        'max_frames': 75,
        'max_sentence_len': 30,
        "lr": 10e-4,
        'epochs': 10000,
        'image_width': 160,
        'image_height': 80,
        'num_workers': 1,
        'n_head': 8,
        'n_layers': 6,
        'd_model': 512,
        'print_loss_every': 10,
        'validation_items': 5,
        'validation_interval': 100,
        'preload': None,
        'weights_folder': 'weights',
        'weights_filename': 'italip_{}.pt',
        'tokenizer_folder': 'tokenizer',
        'tokenizer_filename': 'tokenizer.json'
    }

def get_weights_file(options, epoch: str):
    return Path('.') / options['weights_folder'] / (options['weights_filename'].format(epoch))