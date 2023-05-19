import torch
from fstools import DatasetFSHelper
from textmodel_datasets import ItaLipDataset, ItaLipRawDataset, buildOrLoadTokenizer
from textmodel_model import build_transformer, causal_mask
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
import torchmetrics

def validate(global_step: int):
    model.eval()
    with torch.no_grad():
        dl_iterator = tqdm(range(0, options['validation_items']), desc='Validation')
        itemsCount = 0

        predictedList = []
        targetList = []
        
        for _ in dl_iterator:
            batch = next(iter(val_dl))
            batch_size = batch['encoder_input_ids'].size(0)
            assert batch_size == 1, 'Batch size must be 1 for validation'

            maxSentenceLength = options['max_sentence_len']
            
            # Run the encoder once and save its output
            encoder_input = batch['encoder_input_ids'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_attention_mask'].to(device) # (B, 1, 1, seq_len)
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, dmodel)

            # Initially we start with the SOS token
            decoder_input = torch.tensor([sos_idx]).unsqueeze(0).type_as(batch['decoder_input_ids']).to(device) # (B, 1)
            while True:
                if decoder_input.size(1) == maxSentenceLength:
                    break

                decoder_mask = causal_mask(decoder_input.size(1)).unsqueeze(0).to(device) # (B, 1, seq_len, seq_len)

                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
                nextToken = proj_output[:,-1,:].argmax(dim=1).item()
                
                # Concat new token to previous input ids
                decoder_input = torch.cat([decoder_input, torch.tensor([nextToken]).unsqueeze(0).to(device)], dim=1)

                if nextToken == eos_idx:
                    break
                
            # Convert the predicted ids to words
            predictedList.append(tokenizer.decode(decoder_input[0][1:].cpu().detach().numpy(), skip_special_tokens=False))
            targetList.append(batch['encoder_sentence'][0])
        
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
        writer.add_scalar('val cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predictedList, targetList)
        writer.add_scalar('val wer', wer, global_step)
        writer.flush()


def train():
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
    total_dl_iterations = 0
    for epoch in range(options['epochs']):
        dl_iterator = tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}/{options["epochs"]:02d}')
        for batch in dl_iterator:
            model.train()

            encoder_input = batch['encoder_input_ids'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_attention_mask'].to(device) # (B, 1, 1, seq_len)

            decoder_input = batch['decoder_input_ids'].to(device) # (B, seq_len)
            decoder_mask = batch['decoder_attention_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label_input_ids'].to(device)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, options['vocabulary_size']), label.view(-1))

            # Log the loss
            writer.add_scalar('train loss', loss.item(), total_dl_iterations)
            writer.flush()
            dl_iterator.set_postfix({'loss': f"{loss.item():6.3f}", 'lr': f"{optimizer.param_groups[0]['lr']:6.1e}"})

            # Update the model
            optimizer.step()
            optimizer.zero_grad()

            # Run validation
            if (total_dl_iterations) % options['validation_interval'] == 0:
                validate(total_dl_iterations)

            total_dl_iterations += 1
        # Save the model after each epoch
        torch.save(model.state_dict(), f'./weights/italip_{epoch:02d}.pt')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    options = {
        'batch_size': 12,
        'max_frames': 75,
        'max_sentence_len': 150,
        'lr': 10**-4,
        'epochs': 10000,
        'image_width': 160,
        'image_height': 80,
        'num_workers': 1,
        'nhead': 8,
        'print_loss_every': 10,
        'validation_items': 5,
        'validation_interval': 100,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    tokenizer = buildOrLoadTokenizer('./lipnet_datasets/tokenizer.json')

    raw_ds = ItaLipRawDataset(DatasetFSHelper())

    ds = ItaLipDataset(raw_ds, options['max_frames'], options['max_sentence_len'], tokenizer, imageWidth=options['image_width'], imageHeight=options['image_height'], normalize=True)

    # Get the vocabulary size
    options['vocabulary_size'] = tokenizer.get_vocab_size()
    padding_idx = tokenizer.token_to_id("<PAD>")
    sos_idx = tokenizer.token_to_id("<S>")
    eos_idx = tokenizer.token_to_id("</S>")

    torch.autograd.set_detect_anomaly(True, True)

    # Split into train and validation set with 80% and 20% of the data respectively
    train_ds_len = int(len(ds) * 0.8)
    val_ds_len = len(ds) - train_ds_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_len, val_ds_len])

    # Create the data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'])
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=options['num_workers'])

    # Create the model
    model = build_transformer(src_vocab_size=options['vocabulary_size'], tgt_vocab_size=options['vocabulary_size'], src_seq_len=options['max_sentence_len'], tgt_seq_len=options['max_sentence_len']).to(device)
    train()