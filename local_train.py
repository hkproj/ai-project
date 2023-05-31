if __name__ == '__main__':
    from options import get_config

    config = get_config()
    config['batch_size'] = 5
    config['sleep'] = 0.5
    config['preload'] = '0054'
    config['allow_cuda'] = True
    config['lr'] = (10e-4)/2
    config['weights_folder'] = '/mnt/g/My Drive/Models/ai-project/weights'
    config['tokenizer_folder'] = '/mnt/g/My Drive/Models/ai-project/dataset'
    config['dataset_folder'] = '/mnt/g/My Drive/Models/ai-project/dataset'
    config['experiment_name'] = '/mnt/g/My Drive/Models/ai-project/runs/italip'

    from train import run
    run(config)
