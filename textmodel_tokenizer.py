import tokenizers
import fstools
from pathlib import Path
from tqdm import tqdm
from textmodel_datasets import ItaLipDataset, loadVocabulary
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def getSentenceFromDs(ds) -> str:
    for i in range(len(ds)):
        item = ds[i]
        sentence = item['sentence']
        yield sentence

if __name__ == '__main__':
    # Load the vocabulary file
    tokenizerFilePath = Path('./lipnet_datasets/tokenizer.json')
    vocabularyFile = Path('./lipnet_datasets/vocabulary.txt')
    vocabulary = loadVocabulary(vocabularyFile)
    ds = ItaLipDataset(fstools.DatasetFSHelper(), 75, 200, vocabulary, 160, 80, normalize=True)
    
    if not tokenizerFilePath.exists():
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        trainer = BpeTrainer(special_tokens=["<UNK>", "<S>", "</S>", "<PAD>"])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(getSentenceFromDs(ds), trainer=trainer)
        tokenizer.save(str(tokenizerFilePath))
    
    tokenizer = Tokenizer.from_file(str(tokenizerFilePath))
    padding_idx = tokenizer.token_to_id("<PAD>")
    assert padding_idx == 3, "Padding index is not 3"
    tokenizer.enable_padding(pad_id=padding_idx, pad_token="<PAD>", length=200)
    encoder_tokens = tokenizer.encode("<S>CIAO COME STAI BENISSIMO OTTIMISSIMAMENTE DIREI</S>").tokens
    encoder_input_ids = tokenizer.encode("<S>CIAO COME STAI BENISSIMO OTTIMISSIMAMENTE DIREI</S>").ids
    mask = tokenizer.encode("<S>CIAO COME STAI BENISSIMO OTTIMISSIMAMENTE DIREI</S>").attention_mask
    print(encoder_input_ids)
    print (len(encoder_input_ids))
    
