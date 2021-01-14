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
        return_val = None
        try:
            return_val = torch.from_numpy(x).to(device).float()
        except:
            return_val = torch.from_numpy(x.copy()).to(device).float()

        return return_val