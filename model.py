import torch
import torch.nn as nn
import math
import torchvision
from tqdm import tqdm

# Create a VGG-16 + Transformer as per this paper: https://www.sciencedirect.com/science/article/pii/S187705092200182X

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_key_padding_mask(batch_size, max_seq_size, actual_len):
    src_key_padding_mask = torch.zeros((batch_size, max_seq_size)).type(torch.uint8)
    for b in range(batch_size):
        src_key_padding_mask[b, :actual_len[b]] = 1
    return src_key_padding_mask == 0

def get_causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.uint8)
    return mask == 0

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ItaLipModel(nn.Module):

    def __init__(self, vocab_size, vgg_features_size=1000, src_seq_len=75, tgt_seq_len=200, dropout=0.2, d_model=512, nhead=8, num_encoder_layers=6, dim_ff=2048, padding_idx=0) -> None:
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_ff = dim_ff
        self.padding_idx = padding_idx

        vgg_weights = torchvision.models.VGG11_Weights.DEFAULT
        self.vgg_embedding = torchvision.models.vgg11(weights=vgg_weights)
        self.vgg_features_size = vgg_features_size

        # Convert vgg features into a vector of dmodel size
        self.vgg2embedding = nn.Linear(in_features=vgg_features_size, out_features=d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max(src_seq_len, tgt_seq_len))
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_encoder_layers, dim_feedforward=dim_ff, dropout=dropout, activation='relu')
        self.fc = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        # src:          (B, SEQ_SRC, C, H, W)
        # tgt:          (B, SEQ_TGT)
        # src_mask:     (B, SEQ_SRC)
        # tgt_mask:     (B, SEQ_TGT, SEQ_TGT)

        # For each frame, extract features using VGG
        seq_len_src = src.size(1)
        # Transformer src: (B, SEQ_SRC, D_MODEL)
        input_features = torch.empty((src.size(0), seq_len_src, self.d_model)).type_as(src).to(src.device)
        for t in range(seq_len_src):
            frame_features = self.vgg_embedding(src[:, t, :, :, :]) # (B, 1000)
            input_features[:, t, :] = self.vgg2embedding(frame_features)

        # src: (B, SEQ_SRC, D_MODEL) -> (B, SEQ_SRC, D_MODEL)
        src = input_features
        del input_features
        # src: (B, SEQ_SRC, D_MODEL) -> (B, SEQ_SRC, D_MODEL)
        src = self.positional_encoding(src)
        # tgt: (B, SEQ_TGT) -> (B, SEQ_TGT, D_MODEL)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # tgt: (B, SEQ_TGT, D_MODEL) -> (B, SEQ_TGT, D_MODEL)
        tgt = self.positional_encoding(tgt)
        # Prepare input for transformer
        # src: (SEQ_SRC, B, D_MODEL) -> (SEQ_SRC, B, D_MODEL)
        src = src.permute(1, 0, 2)
        # tgt: (SEQ_TGT, B, D_MODEL) -> (SEQ_TGT, B, D_MODEL)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer(
            src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.fc(output)
        output = torch.log_softmax(output, dim=-1)

        # output: (SEQ_TGT, B, VOCAB_SIZE) -> (B, SEQ_TGT, VOCAB_SIZE)
        output.permute(1, 0, 2)
        return output

if __name__ == '__main__':
    vocab_size = 1000
    batch_size = 2
    src_seq_len = 75
    tgt_seq_len = 200
    image_width = 160
    image_height = 80
    padding_idx = 0

    nhead = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a = ItaLipModel(vocab_size=1000, nhead=nhead).to(device)

    for i in tqdm(range(1000)):

        src = torch.zeros((batch_size, src_seq_len, 3, image_height, image_width)).to(device)
        src_len = torch.zeros((batch_size,)).type(torch.int).to(device)

        tgt = torch.zeros((batch_size, tgt_seq_len)).type(torch.long).to(device)
        tgt_len = torch.zeros((batch_size,)).type(torch.int).to(device)

        for b in range(batch_size):
            num_frames = torch.randint(1, src_seq_len, (1,)).item()
            video = torch.rand((num_frames, 3, image_height, image_width)).cuda().to(device)
            src[b, :num_frames, :, :, :] = video
            src_len[b] = num_frames

            num_words = torch.randint(1, tgt_seq_len, (1,)).item()
            words = torch.randint(0, vocab_size, (num_words,)).cuda().to(device)    
            tgt[b, :num_words] = words
            tgt_len[b] = num_words

        src_key_padding_mask = get_key_padding_mask(batch_size, src_seq_len, src_len).to(device)
        tgt_key_padding_mask = get_key_padding_mask(batch_size, tgt_seq_len, tgt_len).to(device)
        
        src_mask = (torch.ones((batch_size * nhead, src_seq_len, src_seq_len)) == 0).to(device)
        tgt_mask = torch.cat([get_causal_mask(tgt_seq_len) for _ in range(batch_size * nhead)], dim=0).to(device)

        out = a.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
    print(out.shape)