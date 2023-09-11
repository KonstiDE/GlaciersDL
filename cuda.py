import torch

# cuda tests
if __name__ == '__main__':
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())
