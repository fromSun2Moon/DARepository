
from torch.utils.data import Dataset, DataLoader

def print_msg(msg):
    print(msg, "\n")
    print("=" * 50)


class MyDataset(Dataset):
    def __init__(self, df, gt=None, test_mode=False):
        """supervised learning settings"""
        self.test_mode = test_mode
        if test_mode:
            self.df = df.values
        else:
            assert len(df) == len(gt)
            self.df = df.values
            self.gt = gt.values
        
    def __getitem__(self, index):
        if self.test_mode:
            self.x = self.df[index]
            return torch.Tensor(self.x)
        
        self.x = self.df[index]
        self.y = self.gt[index]
        return torch.Tensor(self.x), torch.Tensor(self.y)
        
    def __len__(self):
        return len(self.df)

