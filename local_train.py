from options import get_config
from pathlib import Path
import re

if __name__ == '__main__':

    config = get_config()
    config['batch_size'] = 5
    config['sleep'] = 0
    config['allow_cuda'] = True
    config['lr'] = (10e-4)/2
    config['weights_folder'] = '/mnt/g/My Drive/Models/ai-project/weights'
    config['tokenizer_folder'] = '/mnt/g/My Drive/Models/ai-project/dataset'
    config['dataset_folder'] = '/mnt/g/My Drive/Models/ai-project/dataset'
    config['experiment_name'] = '/mnt/g/My Drive/Models/ai-project/runs/italip'

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

    from train import run
    run(config)
