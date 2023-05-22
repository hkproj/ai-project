import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from fstools import DatasetFSHelper
from datasets import ItaLipDataset, ItaLipRawDataset, buildOrLoadTokenizer
from model import causal_mask, ItaLipModel
from options import get_config, get_weights_file
from tqdm import tqdm
from pathlib import Path
import warnings
import os
# Avoid tokenizers warning for parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def validate(model, val_dl, config, device, tokenizer, sos_idx, eos_idx, writer, global_step: int):
    model.eval()
    with torch.no_grad():
        dl_iterator = tqdm(range(0, config['validation_items']), desc='Validation')
        predictedList = []
        targetList = []
        maxSentenceLength = config['max_sentence_len']

        for _ in dl_iterator:

            batch = next(iter(val_dl))

            frames = batch['frames'].to(device) # (B, seq_len_frames, 3, H, W)
            batch_size = frames.size(0)
            assert batch_size == 1, 'Batch size must be 1 for validation'           
            # Flatten the frames sequence 
            frames = frames.view(-1, *frames.shape[2:]) # (B * seq_len_frames, 3, H, W)
            # Run the frames through the CNN
            frames = model.cnn.forward(frames) # (B * seq_len_frames, dmodel)
            # Make sure the last dimension of the CNN is dmodel of the transformer
            assert frames.size(-1) == config['d_model'], f'CNN output size must be {config["dmodel"]}'
            # Reshape the CNN output to (B, seq_len_frames, dmodel)
            frames = frames.view(batch_size, -1, frames.size(-1)) # (B, seq_len_frames, dmodel)
            encoder_input = frames # (B, seq_len)
            encoder_mask = batch['frames_attention_mask'].to(device) # (B, 1, 1, seq_len)
            encoder_output = model.transformer.encode(encoder_input, encoder_mask)

            # Initially we start with the SOS token
            decoder_input = torch.tensor([sos_idx]).unsqueeze(0).type_as(batch['decoder_input_ids']).to(device) # (B, 1)
            while True:
                if decoder_input.size(1) == maxSentenceLength:
                    break

                decoder_mask = causal_mask(decoder_input.size(1)).unsqueeze(0).to(device) # (B, 1, seq_len, seq_len)
                decoder_output = model.transformer.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.transformer.project(decoder_output) # (B, seq_len, vocab_size)
                nextToken = proj_output[:,-1,:].argmax(dim=1).item()
                
                # Concat new token to previous input ids
                decoder_input = torch.cat([decoder_input, torch.tensor([nextToken]).unsqueeze(0).to(device)], dim=1)

                if nextToken == eos_idx:
                    break
                
            # Convert the predicted ids to words
            predictedList.append(tokenizer.decode(decoder_input[0][1:].cpu().detach().numpy(), skip_special_tokens=False))
            targetList.append(batch['label_sentence'][0])
        
        # Visualize the predicted sentences vs the target sentences
        maxPrintedLength = 80
        print(f'{"-"*maxPrintedLength:<80}|{"-"*maxPrintedLength:<80}')
        print(f'{"PREDICTED":^80}|{"TARGET":^80}')
        print(f'{"-"*maxPrintedLength:<80}|{"-"*maxPrintedLength:<80}')
        for predicted, target in zip(predictedList, targetList):
            print(f'{predicted[:maxPrintedLength]:<80}|{target[:maxPrintedLength]:<80}')
            print(f'{"-"*maxPrintedLength:<80}|{"-"*maxPrintedLength:<80}')

        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predictedList, targetList)
        writer.add_scalar('validation_cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predictedList, targetList)
        writer.add_scalar('validation_wer', wer, global_step)
        writer.flush()

def train(model, train_dl, val_dl, tokenizer, writer, device, config, padding_idx, sos_idx, eos_idx):
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    total_dl_iterations = 0

    start_epoch = 0
    if config['preload'] is not None:
        preload_filepath = get_weights_file(config, config['preload'])
        print(f'Loading weights from {preload_filepath}')
        model.load_state_dict(torch.load(preload_filepath))
        start_epoch = int(config['preload']) + 1

    for epoch in range(start_epoch, config['epochs']):
        dl_iterator = tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}/{config["epochs"]:02d}')
        for batch in dl_iterator:
            model.train()
            torch.cuda.empty_cache()

            frames = batch['frames'].to(device) # (B, seq_len_frames, 3, H, W)

            batch_size = frames.size(0)

            # Flatten the frames sequence 
            frames = frames.view(-1, *frames.shape[2:]) # (B * seq_len_frames, 3, H, W)
            # Run the frames through the CNN
            frames = model.cnn.forward(frames) # (B * seq_len_frames, dmodel)
            # Make sure the last dimension of the CNN is dmodel of the transformer
            assert frames.size(-1) == config['d_model'], f'CNN output size must be {config["dmodel"]}'
            # Reshape the CNN output to (B, seq_len_frames, dmodel)
            frames = frames.view(batch_size, -1, frames.size(-1)) # (B, seq_len_frames, dmodel)
            encoder_input = frames # (B, seq_len)
            encoder_mask = batch['frames_attention_mask'].to(device) # (B, 1, 1, seq_len)

            decoder_input = batch['decoder_input_ids'].to(device) # (B, seq_len)
            decoder_mask = batch['decoder_attention_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.transformer.encode(encoder_input, encoder_mask)
            decoder_output = model.transformer.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.transformer.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label_input_ids'].to(device)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, config['vocabulary_size']), label.view(-1))
            # Compute the gradients
            loss.backward()

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), total_dl_iterations)
            writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], total_dl_iterations)
            writer.flush()
            dl_iterator.set_postfix({'loss': f"{loss.item():6.3f}", 'lr': f"{optimizer.param_groups[0]['lr']:6.1e}"})

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            # Run validation
            if (total_dl_iterations) % config['validation_interval'] == 0:
                validate(model, val_dl, config, device, tokenizer, sos_idx, eos_idx, writer, total_dl_iterations)

            total_dl_iterations += 1
        
        # Save the model after each epoch
        torch.save(model.state_dict(), get_weights_file(config, f"{epoch:02d}"))

def run(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    writer = SummaryWriter()
    tokenizer_file_path = Path('.') / config['tokenizer_folder'] / config['tokenizer_filename']
    tokenizer = buildOrLoadTokenizer(tokenizer_file_path)
    raw_ds = ItaLipRawDataset(DatasetFSHelper())
    ds = ItaLipDataset(raw_ds, config['max_frames'], config['max_sentence_len'], tokenizer, imageWidth=config['image_width'], imageHeight=config['image_height'], normalize=True)

    # Get the vocabulary size
    config['vocabulary_size'] = tokenizer.get_vocab_size()
    print(f'Vocabulary size: {tokenizer.get_vocab_size()}')

    # Save commonly used tokens
    padding_idx = tokenizer.token_to_id("<PAD>")
    sos_idx = tokenizer.token_to_id("<S>")
    eos_idx = tokenizer.token_to_id("</S>")

    torch.autograd.set_detect_anomaly(True, True)

    # Split into train and validation set with 80% and 20% of the data respectively
    train_ds_len = int(len(ds) * 0.8)
    val_ds_len = len(ds) - train_ds_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_len, val_ds_len])

    # Create the data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])

    # Create the model
    
    model = ItaLipModel(src_vocab_size=None, tgt_vocab_size=config['vocabulary_size'], src_seq_len=config['max_frames'], tgt_seq_len=config['max_sentence_len'], d_model=config['d_model'], N=config['n_layers'], h=config['n_head']).to(device)
    train(model, train_dl, val_dl, tokenizer, writer, device, config, padding_idx, sos_idx, eos_idx)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    run(config)

    