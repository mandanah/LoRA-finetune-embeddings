from torch.utils.data import Dataset


class PairDataset(Dataset):
    """A dataset class for handling pairs of data items. Expects a list of tuples, where first element is the anchor and the second is the positive item."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
