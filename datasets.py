from torch.utils.data import Dataset
from pathlib import Path
from filesystem import getVideoId
from log import getLogger

LOG = getLogger(__name__)

class VideoDataset(Dataset):

    def __init__(self, dir: str) -> None:
        super().__init__()
        
        self.ids = []
        path = Path(dir)
        for item in path.iterdir():
            if item.is_file():
                try:
                    self.ids.append(getVideoId(item.name))
                except:
                    LOG.warning(f'Ignoring file {item.name}')
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index) -> str:
        return self.ids[index]
