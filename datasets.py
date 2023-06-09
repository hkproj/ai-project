from torch.utils.data import Dataset
from pathlib import Path
import fstools
from log import getLogger
import os
from torchvision.transforms import transforms
from srtloader import SRTLoader
from PIL import Image
import torch
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import causal_mask

logger = getLogger(os.path.splitext(os.path.basename(__file__))[0])

class VideoDataset(Dataset):

    def __init__(self, rawVideosPath: str) -> None:
        super().__init__()
        
        self.ids = []
        path = Path(rawVideosPath)
        for item in path.iterdir():
            if item.is_file():
                try:
                    self.ids.append(fstools.DatasetFSHelper.getVideoIdFromFileName(item.name))
                except:
                    logger.warning(f'Invalid file name {item}')
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index) -> str:
        return self.ids[index]

class GridRawDataset(Dataset):

    def __init__(self, rootFolder: str) -> None:
        super().__init__()
        self.rootFolder = Path(rootFolder)
        self.lipFolder = self.rootFolder / 'lip'
        self.txtFolder = self.rootFolder / 'GRID_align_txt'

        # Results from the entire dataset
        # Mean: tensor([0.7156, 0.5136, 0.3476], device='cuda:0')
        # Std: tensor([0.1211, 0.1084, 0.0845], device='cuda:0')
        self.mean = [0.7156, 0.5136, 0.3476]
        self.std = [0.1211, 0.1084, 0.0845]

        self.ids = []
        for videoPath in self.lipFolder.iterdir():
            if videoPath.is_dir():
                videoId = videoPath.name
                for miniclipPath in self.getMiniClipFolder(videoPath).iterdir():
                    if miniclipPath.is_dir():
                        miniclipId = miniclipPath.name
                        # Get the number of frames in the folder
                        numFrames = len([f for f in miniclipPath.iterdir() if f.name.endswith(fstools.FRAME_FILE_EXTENSION)])
                        # If the number of frames is 75, add it to the list
                        if numFrames == 75:
                            self.ids.append((videoId, miniclipId))

    def getMiniClipFolder(self, videoPath: Path) -> Path:
        return videoPath / 'video' / 'mpg_6000'
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        videoId, miniclipId = self.ids[index]
        # Get miniclip path
        miniclipPath = self.getMiniClipFolder(self.lipFolder / videoId) / miniclipId
        
        # Get all frames
        framesFilePaths = []
        for framePath in miniclipPath.iterdir():
            if framePath.name.endswith(fstools.FRAME_FILE_EXTENSION):
                framesFilePaths.append(framePath)
        
        # Sort frames by name
        framesFilePaths.sort(key=lambda x: int(x.stem))

        # Load all the frames
        frames = []
        for framePath in framesFilePaths:
            # Load the image with PIL
            img = Image.open(framePath)
            # Add the image to the list
            frames.append((img))

        # Assert that the number of frames = 75
        assert len(frames) == 75, f'Number of frames is not 75, but {len(frames)}'

        transcriptPath = self.txtFolder / videoId / 'align' / (miniclipId + '.align')
        assert transcriptPath.exists(), f'Transcript file {transcriptPath} does not exist'

        fullSentence = ''

        # Load all the lines in the transcript file
        with open(transcriptPath, 'r') as f:
            lines = [l for l in f.readlines() if l.strip() != '']
            # For each line, split it by space
            lines = [l.split(' ') for l in lines]
            for words in lines:
                assert len(words) == 3
                # Get the words
                word = words[2].strip("\r\n").upper()
                if word != 'SIL' and word != 'SP':
                    fullSentence += word + ' '
        
        fullSentence = fullSentence.strip()

        return {
            'videoId': videoId,
            'miniclipId': miniclipId,
            'raw_frames': frames,
            'raw_frames_len': len(frames),
            'raw_sentence': fullSentence,
            'raw_sentence_len': len(fullSentence)
        }

class ItaLipRawDataset(Dataset):

    def __init__(self, dsHelper: fstools.DatasetFSHelper) -> None:
        super().__init__()
        self.miniclipsPath = Path(dsHelper.getMiniClipsFolderPath())

        # Results on the entire dataset:
        # Mean: tensor([0.4076, 0.4406, 0.6105])
        # Std: tensor([0.1354, 0.1512, 0.1707])
        self.mean = [0.4076, 0.4406, 0.6105]
        self.std = [0.1354, 0.1512, 0.1707]

        self.dsHelper = dsHelper
        self.ids = []
        for videoPath in self.miniclipsPath.iterdir():
            if videoPath.is_dir():
                videoId = videoPath.name
                for miniclip in videoPath.iterdir():
                    if miniclip.is_dir():
                        miniclipId = miniclip.name
                        self.ids.append((videoId, miniclipId))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        videoId, miniclipId = self.ids[index]
        # Get miniclip path
        miniclipPath = Path(self.dsHelper.getMiniClipsPath(videoId)) / miniclipId
        # Get lips path
        lipsPath = miniclipPath / fstools.MINI_CLIP_LIPS_DIR
        # Get transcript path
        transcriptPath = miniclipPath / (fstools.MINI_CLIP_TRANSCRIPT_NAME + fstools.TRANSCRIPTION_FILE_EXTENSION)

        # Load the transcript
        srt = SRTLoader(str(transcriptPath))
        words = srt.getAllWords()
        fullSentence = " ".join([w for _, _, _, w in words]).upper()

        # Get all frames
        framesFilePaths = []
        for framePath in lipsPath.iterdir():
            if framePath.name.endswith(fstools.FRAME_FILE_EXTENSION):
                framesFilePaths.append(framePath)
        
        # Sort frames by name
        framesFilePaths.sort(key=lambda x: int(x.stem))

        # Load all the frames
        frames = []
        for framePath in framesFilePaths:
            # Load the image with PIL
            img = Image.open(framePath)
            # Add the image to the list
            frames.append((img))
        
        return {
            'videoId': videoId,
            'miniclipId': miniclipId,
            'raw_frames': frames,
            'raw_frames_len': len(frames),
            'raw_sentence': fullSentence,
            'raw_sentence_len': len(fullSentence)
        }


class ItaLipDataset(Dataset):

    def __init__(self, rawDs: ItaLipRawDataset, max_frames: int, max_sentence_len: int, tokenizer, imageWidth: int, imageHeight: int, normalize: bool = True) -> None:
        super().__init__()
        
        # Create torch transform to load the images
        # Results on the entire dataset:
        # Mean: tensor([0.4076, 0.4406, 0.6105])
        # Std: tensor([0.1354, 0.1512, 0.1707])
        self.transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((imageHeight, imageWidth)),
            transforms.Normalize(
                mean=rawDs.mean,
                std=rawDs.std
            )
        ])

        self.max_frames = max_frames
        self.max_sentence_len = max_sentence_len
        self.rawDs = rawDs

        # Enable padding on the tokenizer
        self.tokenizer = tokenizer
        padding_idx = tokenizer.token_to_id("<PAD>")
        self.tokenizer.enable_padding(length=max_sentence_len, pad_id=padding_idx, padding_token="<PAD>")
        
    def __len__(self):
        return len(self.rawDs.ids)
    
    def __getitem__(self, index) -> dict:
        rawData = self.rawDs[index]

        videoId = rawData['videoId']
        miniclipId = rawData['miniclipId']
        raw_frames = rawData['raw_frames']
        raw_sentence = rawData['raw_sentence']

        # convert list of PIL images into a 4D tensor
        frames = torch.stack([self.transform(frame) for frame in raw_frames])
        # Pad the frames with zeros
        frames_padding = torch.zeros((self.max_frames - len(frames), *frames.shape[1:]))
        frames = torch.cat((frames, frames_padding), 0)
        frames_len = rawData['raw_frames_len']
        frames_attention_mask = torch.tensor([1] * frames_len + [0] * (self.max_frames - frames_len), dtype=torch.long).unsqueeze(0).unsqueeze(0)

        # Convert the sentence into its input ids and tokens
        encoder_sentence = f'<S>{raw_sentence}</S>'
        encoder_sentence_enc = self.tokenizer.encode(encoder_sentence)
        encoder_input_ids = torch.tensor(encoder_sentence_enc.ids, dtype=torch.long)
        encoder_attention_mask = torch.tensor(encoder_sentence_enc.attention_mask, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        encoder_words_len = sum(encoder_sentence_enc.attention_mask)

        # Create input for decoder
        decoder_sentence = f'<S>{raw_sentence}'
        decoder_sentence_enc = self.tokenizer.encode(decoder_sentence)
        decoder_input_ids = torch.tensor(decoder_sentence_enc.ids, dtype=torch.long)
        decoder_words_len = sum(decoder_sentence_enc.attention_mask)
        decoder_attention_mask = torch.tensor(decoder_sentence_enc.attention_mask, dtype=torch.long) & causal_mask(len(decoder_sentence_enc.attention_mask))

        # Create label for decoder
        label_sentence = f'{raw_sentence}</S>'
        label_sentence_enc = self.tokenizer.encode(label_sentence)
        label_input_ids = torch.tensor(label_sentence_enc.ids, dtype=torch.long)
        label_words_len = sum(label_sentence_enc.attention_mask)
        label_attention_mask = torch.tensor(label_sentence_enc.attention_mask, dtype=torch.long) & causal_mask(len(label_sentence_enc.attention_mask))

        # Make sure no sentence is reaching the max sequence length, otherwise some words may have been cut
        assert encoder_words_len < self.max_sentence_len, f"Encoder sentence is too long ({encoder_words_len} >= {self.max_sentence_len})"
        assert decoder_words_len < self.max_sentence_len, f"Decoder sentence is too long ({decoder_words_len} >= {self.max_sentence_len})"
        assert label_words_len < self.max_sentence_len, f"Label sentence is too long ({label_words_len} >= {self.max_sentence_len})"
        
        # Return the dictionary
        return {
            'videoId': videoId,
            'miniclipId': miniclipId,
            'raw_sentence': raw_sentence,
            
            'frames': frames,
            'frames_attention_mask': frames_attention_mask,
            'frames_len': frames_len,

            'encoder_sentence': encoder_sentence, # 
            'encoder_input_ids': encoder_input_ids, # (seq_len)
            'encoder_attention_mask': encoder_attention_mask, # (1, 1, seq_len)
            'encoder_words_len': encoder_words_len,

            'decoder_sentence': decoder_sentence,
            'decoder_input_ids': decoder_input_ids, # (seq_len)
            'decoder_attention_mask': decoder_attention_mask, # (1, 1, seq_len)
            'decoder_words_len': decoder_words_len,

            'label_sentence': label_sentence,
            'label_input_ids': label_input_ids, # (seq_len)
            'label_attention_mask': label_attention_mask, # (1, 1, seq_len)
            'label_words_len': label_words_len
        }

def getSentenceFromDs(ds: ItaLipRawDataset) -> str:
    for i in range(len(ds)):
        item = ds[i]
        sentence = item['raw_sentence']
        yield sentence

def buildOrLoadTokenizer(raw_ds: Dataset, filePath: set) -> Tokenizer:
    tokenizerFilePath = Path(filePath)
    if not tokenizerFilePath.exists():
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        trainer = BpeTrainer(special_tokens=["<UNK>", "<S>", "</S>", "<PAD>"])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(getSentenceFromDs(raw_ds), trainer=trainer)
        tokenizer.save(str(tokenizerFilePath))
    
    tokenizer = Tokenizer.from_file(str(tokenizerFilePath))
    return tokenizer

def calculateMeanAndStd(ds: Dataset):
    # Calculate the mean and std of all frames in the ds along the RGB channels
    mean = 0
    std = 0
    
    # Use the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = transforms.ToTensor()

    for i in tqdm(range(len(ds))):
        frames = ds[i]['raw_frames']
        # convert list of PIL images into a 4D tensor
        frames = torch.stack([t(frame) for frame in frames]).to(device)
        mean += frames.mean(dim=(0, 2, 3))
        std += frames.std(dim=(0, 2, 3))
    mean /= len(ds)
    std /= len(ds)

    print(f'Mean: {mean}')
    print(f'Std: {std}')

if __name__ == '__main__':
    calculateMeanAndStd(GridRawDataset(str(Path('GRID_LIP'))))
    
    