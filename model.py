from tinygrad.tensor import Tensor
from tinygrad import nn

def preprocess(x: Tensor) -> Tensor:
  x = x.float() / 255
  return x.permute(0, 3, 1, 2)

# Model0

# Model1

# Model2
