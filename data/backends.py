# backends.py updates


class NumpyBackend:
    import numpy as np
    @staticmethod
    def stack(arrays): return np.stack(arrays)
    @staticmethod
    def convert(data): return np.array(data, dtype=np.float32)

class JaxBackend:
    def __init__(self):
        import jax.numpy as jnp
        self.jnp = jnp
    def stack(self, arrays): return self.jnp.stack(arrays)
    def convert(self, data): return self.jnp.array(data)

class TorchBackend:
    def __init__(self):
        import torch
        self.torch = torch
    def stack(self, tensors): return self.torch.stack(tensors)
    def convert(self, data): return self.torch.tensor(data, dtype=self.torch.float32)