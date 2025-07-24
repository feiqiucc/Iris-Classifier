import numpy as np
import h5py
import math
import struct
import matplotlib.pyplot as plt
from data_process import *
from activation_funcs import *
from initialization import *
from model import *
from evaluate import evaluate

plt.ion()
fig, ax = plt.subplots()
