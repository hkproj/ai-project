from options import get_config
from pathlib import Path
import re
from train import run

if __name__ == '__main__':
    config = get_config()
    config['batch_size'] = 12
    config['sleep'] = 0
    config['lr'] = 10**-2
    config['allow_cuda'] = True
    config['dataset_type'] = 'GRID'
    config['GRID_root'] = 'GRID_LIP'
    config['weights_folder'] = 'GRID_LIP/weights'
    config['tokenizer_folder'] = 'GRID_LIP/dataset'
    config['dataset_folder'] = 'GRID_LIP/dataset'
    config['experiment_name'] = 'GRID_LIP/runs/GRID'
    config['n_layers'] = 1
    config['n_head'] = 1
    config['max_sentence_len'] = 15

    maxEpoch = -1
    for filename in Path(config['weights_folder']).glob('**/*.pt'):
        match = re.search(r'\d+', filename.name)
        if match:
            epoch = int(match.group())
            maxEpoch = max(maxEpoch, epoch)
        else:
            raise ValueError(f'No epoch number found in filename: {filename}')

    maxEpochStr = f'{maxEpoch:04d}' if maxEpoch >= 0 else None
    config['preload'] = maxEpochStr

    run(config)
