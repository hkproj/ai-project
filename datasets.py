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

class ItaLipDataset(Dataset):

    def __init__(self, dsHelper: fstools.DatasetFSHelper, max_frames: int, max_sentence_len: int, vocabulary: dict, imageWidth: int, imageHeight: int, normalize: bool = True) -> None:
        super().__init__()
        self.miniclipsPath = Path(dsHelper.getMiniClipsFolderPath())
        
        # Create torch transform to load the images

        t = [transforms.ToTensor(),
            transforms.Resize((imageHeight, imageWidth))
        ]

        # Results on the entire dataset:
        # Mean: tensor([0.4076, 0.4406, 0.6105])
        # Std: tensor([0.1354, 0.1512, 0.1707])

        if normalize:
            t.append(transforms.Normalize(
                mean=[0.4076, 0.4406, 0.6105],
                std=[0.1354,0.1512,0.1707]
            ))
        
        self.transform = transforms.Compose(t)
        self.vocabulary = vocabulary
        self.max_frames = max_frames
        self.max_sentence_len = max_sentence_len

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
    
    def __getitem__(self, index) -> dict:
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

        # Get the padding index 
        padding_idx = self.vocabulary['<PAD>']

        # Get the input ids
        input_ids = self.getInputIds(fullSentence)
        # Concat the <SOS> and <EOS> chars
        input_ids = torch.cat((torch.tensor([self.vocabulary['<S>']]).long(), input_ids, torch.tensor([self.vocabulary['</S>']]).long()), 0)
        # Check that the size is still below the max sentence len
        if len(input_ids) > self.max_sentence_len:
            raise ValueError(f'Sentence too long:  {fullSentence}')
        
        decoder_input_ids = input_ids[:-1]
        target_input_ids = input_ids[1:]

        decoder_input_ids = self.padInputIds(decoder_input_ids, self.max_sentence_len, padding_idx)
        target_input_ids = self.padInputIds(target_input_ids, self.max_sentence_len, padding_idx)



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
            # Apply the transform
            img = self.transform(img)
            # Add the image to the list
            frames.append(img)

        # Pad the frames
        frames = torch.stack(frames)        
        imgShape = frames[0].size()
        frames_padding = torch.zeros((self.max_frames - len(frames), imgShape[0], imgShape[1], imgShape[2]))
        frames = torch.cat((frames, frames_padding), 0)
        
        # Return the dictionary
        return {
            'videoId': videoId,
            'miniclipId': miniclipId,
            'frames': frames,
            'decoder_input_ids': decoder_input_ids,
            'target_input_ids': target_input_ids,
            'input_ids_len': len(fullSentence) + 1, # + 1 because of the <S> or </S> char in the input ids
            'frames_len': len(framesFilePaths),
            'sentence': fullSentence,
        }
    
    def getInputIds(self, text: str) -> torch.Tensor:
        inputIds = []
        for c in text:
            if c == ' ':
                inputIds.append(self.vocabulary['<BLANK>'])
            else:
                inputIds.append(self.vocabulary[c])
        return torch.tensor(inputIds).long()
    
    def padInputIds(self, inputIds: torch.Tensor, seqLen: int, paddingIdx: int) -> torch.Tensor:
        inputIdsPadding = torch.empty(seqLen - len(inputIds)).fill_(paddingIdx).long()
        return torch.cat((inputIds, inputIdsPadding), 0)


def loadVocabulary(filePath: str) -> dict:
    vocabulary = {}
    # Read all the lines
    with open(filePath, 'r') as f:
        # Read all the non-whitespace lines
        chars = set([line.strip() for line in f.read().splitlines() if len(line.strip()) > 0])

    if '\n' in chars:
        chars.remove('\n')
    if '\r' in chars:
        chars.remove('\r')
    
    # Verify that all chars are strings of length 1
    for c in chars:
        assert len(c) == 1, 'Invalid character in vocabulary: \'{}\''.format(c)

    # Sort all the characters
    chars = sorted(list(chars))

    # Add the additional characters
    additional_chars = ['<PAD>', '<BLANK>', '<S>', '</S>']
    chars = additional_chars + chars

    # Create the vocabulary
    vocabulary = {c: i for i, c in enumerate(chars)}
    return vocabulary

def calculateMeanAndStd():
    # Load the vocabulary file
    vocabularyFile = Path('./lipnet_datasets/vocabulary.txt')
    vocabulary = loadVocabulary(vocabularyFile)
    ds = ItaLipDataset(fstools.DatasetFSHelper(), 75, 200, vocabulary, 160, 80, normalize=True)

    # Calculate the mean and std of all frames in the ds along the RGB channels
    mean = 0
    std = 0
    
    # Use the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in tqdm(range(len(ds))):
        frames = ds[i]['frames'].to(device)
        frames_len = ds[i]['frames_len']
        frames = frames[0:frames_len]
        mean += frames.mean(dim=(0, 2, 3))
        std += frames.std(dim=(0, 2, 3))
    mean /= len(ds)
    std /= len(ds)

    print(f'Mean: {mean}')
    print(f'Std: {std}')

if __name__ == '__main__':
    pass
    