import torch
from fstools import DatasetFSHelper
from textmodel_datasets import ItaLipDataset, ItaLipRawDataset, buildOrLoadTokenizer
from textmodel_model import ItaLipModel, get_causal_mask, get_key_padding_mask
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
import torchmetrics

if __name__ == '__main__':
    options = {
        'batch_size': 20,
        'max_frames': 75,
        'max_sentence_len': 200,
        'lr': 10**-4,
        'epochs': 10000,
        'image_width': 160,
        'image_height': 80,
        'num_workers': 4,
        'nhead': 8,
        'print_loss_every': 10,
        'validation_items': 5,
        'validation_interval': 600
    }
    options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    tokenizer = buildOrLoadTokenizer('./lipnet_datasets/tokenizer.json')
    raw_ds = ItaLipRawDataset(DatasetFSHelper())
    ds = ItaLipDataset(raw_ds, options['max_frames'], options['max_sentence_len'], tokenizer, imageWidth=options['image_width'], imageHeight=options['image_height'], normalize=True)
    # Get the vocabulary size
    vocabulary = tokenizer.get_vocab()
    options['vocabulary_size'] = len(vocabulary)
    padding_idx = tokenizer.token_to_id("<PAD>")
    sos_idx = tokenizer.token_to_id("<S>")
    eos_idx = tokenizer.token_to_id("</S>")

def getValidationInputFromBatch(batch: dict, startToken: int, previousInputIds: torch.Tensor) -> tuple:
    # Encoder input
    src = batch['encoder_input_ids'].to(options['device'])
    src_act_seq_len = batch['encoder_input_len']
    src_max_seq_len = options['max_sentence_len']

    # We start with the <S> token and then we append the previous input ids
    tgt = torch.cat([torch.tensor([startToken]).to(options['device']), previousInputIds], dim=0).unsqueeze(0).type_as(batch['decoder_input_ids']).to(options['device'])
    tgt_act_seq_len = torch.tensor([tgt.size(1)]).type_as(batch['decoder_input_len']).to(options['device'])
    tgt_max_seq_len = tgt.size(1)

    batch_size = src.size(0)    

    # Build padding masks
    src_key_padding_mask = get_key_padding_mask(batch_size, src_max_seq_len, src_act_seq_len).to(options['device'])
    tgt_key_padding_mask = get_key_padding_mask(batch_size, tgt_max_seq_len, tgt_act_seq_len).to(options['device'])

    # Build causal masks
    nhead = options['nhead']            
    src_mask = (torch.ones((batch_size * nhead, src_max_seq_len, src_max_seq_len)) == 0).to(options['device'])
    tgt_mask = torch.cat([get_causal_mask(tgt_max_seq_len) for _ in range(batch_size * nhead)], dim=0).to(options['device'])

    return (src, tgt, None, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)


def getTrainingInputFromBatch(batch: dict) -> tuple:
    # Encoder input
    src = batch['encoder_input_ids'].to(options['device'])
    src_act_seq_len = batch['encoder_input_len']
    src_max_seq_len = options['max_sentence_len']
    
    # Decoder input
    tgt = batch['decoder_input_ids'].to(options['device'])
    tgt_act_seq_len = batch['decoder_input_len']
    tgt_max_seq_len = options['max_sentence_len']
    
    # Labels
    lbl = batch['label_input_ids'].to(options['device'])

    batch_size = src.size(0)

    # Build padding masks
    src_key_padding_mask = get_key_padding_mask(batch_size, src_max_seq_len, src_act_seq_len).to(options['device'])
    tgt_key_padding_mask = get_key_padding_mask(batch_size, tgt_max_seq_len, tgt_act_seq_len).to(options['device'])

    # Build causal masks
    nhead = options['nhead']            
    src_mask = (torch.ones((batch_size * nhead, src_max_seq_len, src_max_seq_len)) == 0).to(options['device'])
    tgt_mask = torch.cat([get_causal_mask(tgt_max_seq_len) for _ in range(batch_size * nhead)], dim=0).to(options['device'])

    return (src, tgt, lbl, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

def validate(val_dl, vocabulary: dict, model: ItaLipModel, global_step: int):
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
            
            # Run a greedy algorithm to predict the sentence
            stop = False
            previousInputIds = torch.tensor([]).type_as(batch['decoder_input_ids']).to(options['device'])
            while True:
                if previousInputIds.size(0) == maxSentenceLength:
                    break

                src, tgt, _, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = getValidationInputFromBatch(batch, sos_idx, previousInputIds)
                output = model.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
                output = output.squeeze(0)
                _, nextToken = torch.max(output, dim=1)
                nextToken = nextToken[-1].item()
                
                # Concat new token to previous input ids
                previousInputIds = torch.cat([previousInputIds, torch.tensor([nextToken]).to(options['device'])], dim=0)

                if nextToken == eos_idx:
                    break
                
            # Convert the predicted ids to words
            predictedList.append(tokenizer.decode(previousInputIds.cpu().detach().numpy(), skip_special_tokens=False))
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


def train(train_dl: DataLoader, val_dl: DataLoader, model: ItaLipModel):
    loss = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
    total_dl_iterations = 0
    for epoch in range(options['epochs']):
        dl_iterator = tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}/{options["epochs"]:02d}')
        for batch in dl_iterator:
            model.train()
            src, tgt, lbl, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = getTrainingInputFromBatch(batch)
            # Forward
            output = model.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

            # Output: (B, SEQ_TGT, VOCAB_SIZE) --> (B * SEQ_TGT, VOCAB_SIZE)
            output = output.reshape(-1, len(vocabulary))
            # Target: (B, SEQ_TGT) --> (B * SEQ_TGT)
            lbl = lbl.view(-1)

            # Compute loss
            loss_val = loss(output, lbl)
            loss_val.backward()

            # Log the loss
            writer.add_scalar('train loss', loss_val.item(), total_dl_iterations)
            writer.flush()
            dl_iterator.set_postfix({'loss': loss_val.item(), 'lr': optimizer.param_groups[0]['lr']})

            # Update the model
            optimizer.step()
            optimizer.zero_grad()

            # Run validation
            if (total_dl_iterations * options['batch_size']) % options['validation_interval'] == 0:
                validate(val_dl, vocabulary, model, total_dl_iterations)

            total_dl_iterations += 1
        # Save the model after each epoch
        torch.save(model.state_dict(), f'./weights/italip_{epoch:02d}.pt')

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True, True)

    # Split into train and validation set with 80% and 20% of the data respectively
    train_ds_len = int(len(ds) * 0.8)
    val_ds_len = len(ds) - train_ds_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_len, val_ds_len])

    # Create the data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'])
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=options['num_workers'])

    # Create the model
    model = ItaLipModel(options['vocabulary_size'], src_seq_len=options['max_sentence_len'], tgt_seq_len=options['max_sentence_len'], padding_idx=padding_idx, nhead=options['nhead']).to(options['device'])
    train(train_dl, val_dl, model)