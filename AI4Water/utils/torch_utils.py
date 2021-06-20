
from torch.utils.data import Dataset

class TorchDataset(Dataset):

    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self.y is None:
            return self.x[item]
        return self.x[item], self.y[item]


def to_torch_dataset(x, y=None):
    return TorchDataset(x, y=y)