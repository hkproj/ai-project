import torch
from fstools import DatasetFSHelper
from datasets import ItaLipDataset, loadVocabulary
from model import ItaLipModel, get_causal_mask, get_key_padding_mask
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings

if __name__ == '__main__':
    options = {
        'batch_size': 8,
        'max_frames': 75,
        'max_sentence_len': 200,
        'lr': 10**-4,
        'epochs': 100,
        'image_width': 160,
        'image_height': 80,
        'num_workers': 4,
        'nhead': 8,
        'print_loss_every': 10,
    }
    writer = SummaryWriter()

def train(train_dl: DataLoader, val_dl: DataLoader, vocabulary: dict, model: ItaLipModel, options: dict):
    loss = torch.nn.CrossEntropyLoss(ignore_index=options['padding_idx'])
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
    model.train()
    total_iterations = 0
    for epoch in range(options['epochs']):
        train_dl_iterator = tqdm(train_dl, desc=f'Epoch {epoch + 1:02d}/{options["epochs"]:02d}')
        for batch in train_dl_iterator:

            src = batch['frames'].to(options['device'])
            tgt = batch['decoder_input_ids'].to(options['device'])
            lbl = batch['target_input_ids'].to(options['device'])
            src_len = batch['frames_len']
            tgt_len = batch['input_ids_len']
            tgt_sentence = batch['sentence']

            # src:          (B, SEQ_SRC, C, H, W)
            # tgt:          (B, SEQ_TGT)
            # src_mask:     (B, SEQ_SRC)
            # tgt_mask:     (B, SEQ_TGT, SEQ_TGT)

            batch_size = src.size(0)
            src_seq_len = options['max_frames']
            tgt_seq_len = options['max_sentence_len']

            src_key_padding_mask = get_key_padding_mask(batch_size, src_seq_len, src_len).to(options['device'])
            tgt_key_padding_mask = get_key_padding_mask(batch_size, tgt_seq_len, tgt_len).to(options['device'])

            nhead = options['nhead']            
            src_mask = (torch.ones((batch_size * nhead, src_seq_len, src_seq_len)) == 0).to(options['device'])
            tgt_mask = torch.cat([get_causal_mask(tgt_seq_len) for _ in range(batch_size * nhead)], dim=0).to(options['device'])

            output = model.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
            # Output: (B, SEQ_TGT, VOCAB_SIZE) --> (B * SEQ_TGT, VOCAB_SIZE)
            assert output.size(2) == len(vocabulary)
            output = output.reshape(-1, len(vocabulary))
            # Target: (B, SEQ_TGT) --> (B * SEQ_TGT)
            lbl = lbl.view(-1)
            loss_val = loss(output, lbl)
            loss_val.backward()

            writer.add_scalar('train loss', loss_val.item(), total_iterations)
            writer.flush()

            train_dl_iterator.set_postfix({'loss': loss_val.item()})

            optimizer.step()
            optimizer.zero_grad()
            total_iterations += 1
        # Save the model after each epoch
        torch.save(model.state_dict(), f'./weights/italip_{epoch:02d}.pt')

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True, True)
    
    options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocabulary = loadVocabulary('./lipnet_datasets/vocabulary.txt')
    options['padding_idx'] = vocabulary['<PAD>']

    ds = ItaLipDataset(DatasetFSHelper(), options['max_frames'], options['max_sentence_len'], vocabulary, imageWidth=options['image_width'], imageHeight=options['image_height'], normalize=False)
    # Split into train and validation set with 80% and 20% of the data respectively
    train_ds_len = int(len(ds) * 0.8)
    val_ds_len = len(ds) - train_ds_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_len, val_ds_len])

    # Create the data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'])
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'])

    # Create the model
    model = ItaLipModel(len(vocabulary), src_seq_len=options['max_frames'], tgt_seq_len=options['max_sentence_len'], padding_idx=options['padding_idx'], nhead=options['nhead']).to(options['device'])
    train(train_dl, val_dl, vocabulary, model, options)
    print('Everything is ready')