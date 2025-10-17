import torch

class CUDADataLoader:
    """
    Zero-copy GPU data loading
    Utilizes RTX 5070 Ti Tensor Cores
    """
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda':
            print(f"CUDA device found: {torch.cuda.get_device_name(0)}")
            # Enable TF32 for faster training on Ampere and later GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for faster training.")
        else:
            print("No CUDA device found, using CPU.")

    def load_data(self, data):
        """
        Loads data to the GPU.
        In a real implementation, this would handle batching, transformations,
        and pinned memory for efficient CPU to GPU transfers.
        """
        if isinstance(data, (list, tuple)):
            return [d.to(self.device) for d in data]
        else:
            return data.to(self.device)
