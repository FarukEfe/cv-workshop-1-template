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
# 3 -> 16 -> 32 -> 64 -> 128
class Model1:
  def __init__(self):
    self.c1 = nn.Conv2d(3, 16, kernel_size=3)
    self.c2 = nn.Conv2d(16, 32, kernel_size=3)
    self.c3 = nn.Conv2d(32, 64, kernel_size=3)
    self.c4 = nn.Conv2d(64, 128, kernel_size=3)
    self.l1 = nn.Linear(128,2)
  
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
    x = x.mean([2, 3])

    x = self.l1(x)
    return x

# Model2

# Batch normalization is a great tool for regularization, and can single handedly increase the model accuracy by
# a huge margin. However it's introducing a new layer, so might slow down training.
class Model2:
  def __init__(self):
    self.c1 = nn.Conv2d(3, 16, kernel_size=3)
    self.n1 = nn.BatchNorm2d(16)
    self.c2 = nn.Conv2d(16, 32, kernel_size=3)
    self.n2 = nn.BatchNorm2d(32)
    self.c3 = nn.Conv2d(32, 64, kernel_size=3)
    self.n3 = nn.BatchNorm2d(64)
    self.c4 = nn.Conv2d(64, 128, kernel_size=3)
    self.n4 = nn.BatchNorm2d(128)
    self.l1 = nn.Linear(128,2)
  
  def __call__(self, x) -> Tensor:
    x = preprocess(x)

    x = x.avg_pool2d(2)
    x = self.n1(self.c1(x))
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.n2(self.c2(x))
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.n3(self.c3(x))
    x = x.gelu()

    x = x.avg_pool2d(2)
    x = self.n4(self.c4(x))
    x = x.mean([2, 3]) # What do we exactly do at this step?

    x = self.l1(x)
    return x

# The ideal batch size is as large as you can go without the computer shitting itself. 
# Since the model trains one batch each step, larger batch means your model will see more examples in total.
# the idea is simple: total examples trained = batch_size * n_steps

# Information Theory: Studies how to persist and manipulate information. How does information maintain?
# What causes loss of information? For example, the reason why we increase the number of filters in each
# layer of convolution is to prevent information loss; the information accumulated from the previous layer
# can be channeled usefully, thus increasing # filters each convolution layer prevents loss of information
# due to reduced dimensions.

# Rectified Linear Unit is great for maximizing features, however it has one issue. Driving all the negative
# weights to 0 causes loss of information, and in practice could give us a useless layer if most of its weights
# are negative. To prevent this, there's another activation function called 'gelu' which stands for 'Gaussian
# Error Linear Unit'.

# Backprop: Instead of using chain rules to derive how to error cost function is going to reduce, we can 
# project a value and skip in-between steps. This is especially handy when you have to deal with 
# non-differentiable error cost functions.

# At Mac RoboMaster we're going to use regressions models for computer vision since we care about detecting
# the center of the armor plate instead of seeing if it is in the frame or not.