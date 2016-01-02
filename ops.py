import numpy as np

def normalize(array, lower, upper):
  maxval = np.amax(array)
  minval = np.amin(array)
  translated = list(map(lambda x: x-minval+lower, array))
  scaled = list(map(lambda x: np.uint8(x*(upper-lower)/(maxval-minval)), translated))
  return np.array(scaled)
