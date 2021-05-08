import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
v = torch.rand(10).to(device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**1,1), 'KB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**1,1), 'KB')
v += 1
print('Done!')