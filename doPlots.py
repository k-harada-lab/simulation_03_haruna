import numpy as np
from tqdm import tqdm
import pandas as pd
import _pickle as pickle

from pprint import pprint
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import seaborn as sns
plt.style.use("./files/paper.mplstyle")

from model import *
from utils import *

np.random.seed(12345)


# Define the parameters
state = {"n_f" : 492, "n_p" : 4, "n_m" : 4, "p_f" : 10, "p" : 10,}
params = {"sigma_eps" : 0.005, "sigma_mu" : 0.05, "t_c": 0.001, "gamma" : 0.01, 
          "beta": 4, "R": 0.0004, "s": 0.75, "alpha_1": 0.6, "alpha_2": 1.5, 
          "alpha_3": 1, "v_1": 2, "v_2": 0.6, "dt": 0.002}

# Load the data from simul0
with open("./files/tmp/simul20251223_141618.pkl", "rb") as f:
    history = pickle.load(f)
totalT = len(history["prices"])
time = np.arange(0, totalT, 1)
args = state, params, time

# Plot for simul0
plots = [
    plotTS(history, args),
    plotLogReturns(history, args),
    plotPopulation(history, args),
    animateXZ(history, args)
]

for i in range(len(plots)):
    plots[i]


# Load the data from simul1 (longer)
with open("./files/simul1.pkl", "rb") as f:
    history = pickle.load(f)
totalT = len(history["prices"])
time = np.arange(0, totalT, 1)
args = state, params, time

# Plot for simul1
plots = [
    plotACF(history, args),
    plotECDF(history, args),
    plotDFA(history, args),
]

for i in range(len(plots)):
    plots[i]




