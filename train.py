import glob, random
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad import TinyJit, dtypes, GlobalCounters
from tinygrad.helpers import tqdm
import cv2
import numpy as np

from common import start_dataloader, stop_dataloader
from model import Model0, Model1, Model2

# hyperparameters
BS = 32
LR = 1e-5
TRAIN_STEPS = 200
TEST_STEPS = 200

# dataloader
def get_train_files():
  return glob.glob(str(Path(__file__).parent / "data" / "train" / "*.txt"))[200:]
def get_test_files():
  return glob.glob(str(Path(__file__).parent / "data" / "train" / "*.txt"))[:200]

def load_single_file(file):
  # read the image file
  img = cv2.imread(file.replace(".txt", ".png"))
  # read the annotation file
  with open(file, "r") as f:
    detected = int(f.readline())
  return img, detected

# train step
@TinyJit
def train_step(x, y):
  pred = model(x)
  loss = loss_fn(pred, y)

  optim.zero_grad()
  loss.backward()
  optim.step()

  return loss.float().realize()

# loss function
def loss_fn(pred: Tensor, y: Tensor):
  return pred.sparse_categorical_crossentropy(y)

# prediction step
@TinyJit
def pred_step(x):
  return model(x)

# main
if __name__ == "__main__":
  dtypes.default_float = dtypes.float32

  # --- TRAINING ---
  # enable training mode
  Tensor.no_grad = False
  Tensor.training = True

  # create model
  # model = Model0()
  # model = Model1()
  model = Model1()

  # create optimizer
  optim = AdamW(get_parameters(model), lr=LR)

  # start dataloder
  data_queue = start_dataloader(get_train_files, load_single_file, BS)

  # train
  for step in (t := tqdm(range(TRAIN_STEPS))):
    GlobalCounters.reset()

    # get a batch of data
    x, y = data_queue.get()
    x = Tensor(np.array(x), dtype=dtypes.uint8)
    y = Tensor(np.array(y), dtype=dtypes.int32)

    # train step
    loss = train_step(x, y)

    # logging
    t.set_description(f"loss: {loss.item():.4f}")

  # stop dataloader
  stop_dataloader()

  print("training done, testing...")

  # start dataloader
  data_queue = start_dataloader(get_test_files, load_single_file, 1)

  # --- TESTING ---
  # enable evaluation mode
  Tensor.no_grad = True
  Tensor.training = False
  random.seed(130)

  correct, total = 0, 0
  for step in (t := tqdm(range(TEST_STEPS))):
    # load a single file
    img, detected = data_queue.get()

    # tensor
    x = Tensor(img, dtype=dtypes.uint8)

    # predict
    pred = pred_step(x)
    pred = pred.argmax().item()

    # compare
    if pred == detected[0]:
      correct += 1
    total += 1

  print(f"accuracy: {correct / total:.4f}")

  # stop dataloader
  stop_dataloader()
