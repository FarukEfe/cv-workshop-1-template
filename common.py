import multiprocessing, random, signal, sys, functools, os
from multiprocessing import Process, Queue, Value
from typing import Callable

import numpy as np

# dataloader common
def try_wrapped_fn(fn, file):
  try:
    return fn(file)
  except:
    return None

def minibatch_iterator(stopped, q: Queue, files_fn: Callable, load_single_file: Callable, bs: int):
  files = files_fn()

  pool = multiprocessing.Pool(4)
  while not stopped.value:
    random.shuffle(files)
    for i in range(0, len(files) - bs, bs):
      if stopped.value:
        break

      batched = pool.map(functools.partial(try_wrapped_fn, load_single_file), files[i : i + bs])
      x_b, y_b = zip(*batched)

      while not stopped.value:
        try:
          q.put((np.array(x_b), np.array(y_b)), block=False)
          break
        except:
          if stopped.value:
            break
      if stopped.value:
        break
  pool.close()
  os._exit(0)

def start_dataloader(files_fn: Callable, load_single_file: Callable, bs: int):
  global bi, stopped
  stopped = Value('b', False)

  # start batch iterator in a separate process
  data_queue = Queue(4)
  bi = Process(target=minibatch_iterator, args=(stopped, data_queue, files_fn, load_single_file, bs))
  bi.start()

  def sigint_handler(*_):
    print("SIGINT received, killing batch iterator")
    stopped.value = True
    bi.kill()
    bi.join()
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  return data_queue

def stop_dataloader():
  stopped.value = True
  bi.join()
