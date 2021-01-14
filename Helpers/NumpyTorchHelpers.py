import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x

def to_torch(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x).to(device).float()