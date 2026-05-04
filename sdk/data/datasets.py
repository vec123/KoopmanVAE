import torch
from torch.utils.data import Dataset

class KoopmanDataset(Dataset):
    def __init__(self, x_data, u_data, f_data, window_size, stride=1):
        """
        x_data: [Total_Len, State_Dim]
        u_data: [Total_Len, Control_Dim]
        f_data: [Total_Len, Forcing_Dim]
        """
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.u = torch.tensor(u_data, dtype=torch.float32)
        self.f = torch.tensor(f_data, dtype=torch.float32)
        self.window_size = window_size
        
        # Calculate valid start indices
        self.indices = np.arange(0, len(x_data) - window_size, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        
        return self.x[start:end], self.u[start:end], self.f[start:end]