from tinygrad.tensor import Tensor
from tinygrad import nn

# neural networks behave better in the 0 - 1 range since there's more floating point precision.
# this is the main reason why we apply normalization while preprocessing
def preprocess(x: Tensor) -> Tensor:
  x = x.float() / 255
  return x.permute(0, 3, 1, 2)

# Images usually come in width x height x 3, however neural networks perform faster when
# they're fed 3 x width x height due to some GPU and caching properties.
# Rule of thumb: use 3 x width x height matrix representation for faster performing neural network models.

# Model0
class Model0:
  def __init__(self):
    self.l1 = nn.Linear(9408, 2)
  
  def __call__(self, x) -> Tensor:
    x = preprocess(x)
    x = x.avg_pool2d(4)
    x = x.flatten(1)
    x = self.l1(x)
    return x

# Model1
class Model1:
  def __init__(self):
    self.c1 = nn.Conv2d()
    self.c2 = nn.Conv2d()
    self.c3 = nn.Conv2d()
    self.c4 = nn.Conv2d()
  
  def __call__(self, x) -> Tensor:
    x = preprocess(x)

    x = x.avg_pool2d(2)
    x = self.c1(x)
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.c2(x)
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.c3(x)
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.c4(x)
    x = x.gelu()

    x = self.l1(x)
    return x


# Model2
