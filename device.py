from torch.cuda import is_available
from torch import device
def dev():
    #device_id = torch.cuda.device_count()
    #device = torch.cuda.get_device_name(range(device_id))
    return device("cuda:0" if is_available() else "cpu")