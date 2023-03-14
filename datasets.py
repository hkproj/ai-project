from torch.utils.data import Dataset
from pathlib import Path
import fstools
from log import getLogger
import os

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
