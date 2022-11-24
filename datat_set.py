from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class AugmentedDataset(Dataset):

    def __init__(self, src_ds: Dataset, transforms=None) -> None:
        super().__init__()
        self.src_ds = src_ds
        self.transforms = transforms

    def __getitem__(self, index) -> T_co:
        x, y = self.src_ds.__getitem__(index)
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return self.src_ds.__len__()
