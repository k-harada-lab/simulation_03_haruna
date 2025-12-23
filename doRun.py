import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import _pickle as pickle
from model import *
from utils import *
from datetime import datetime
import networkx as nx

np.random.seed(12345)

state = {"n_f" : 44, "n_p" : 3, "n_m" : 3, "p_f" : 10, "p" : 10,}
params = {"sigma_eps" : 0.005, "sigma_mu" : 0.05, "t_c": 0.001, "gamma" : 0.01, 
          "beta": 4, "R": 0.0004, "s": 0.75, "alpha_1": 0.6, "alpha_2": 1.5, 
          "alpha_3": 1, "v_1": 2, "v_2": 0.6, "dt": 0.002}
totalT = 30

N = 50  # ノード数
# 全結合
G_full = nx.complete_graph(N)
# スケールフリー
G_sf = nx.barabasi_albert_graph(N, 3)
# 格子状（2D）
G_grid = nx.grid_2d_graph(int(N**0.5), int(N**0.5))
# 木構造（バランス木）
G_tree = nx.balanced_tree(r=2, h=int(np.log2(N)))

agents = []
types = ["p"]*state["n_p"] + ["m"]*state["n_m"] + ["f"]*state["n_f"]
np.random.shuffle(types)
for i in range(N):
    agents.append(Agent(i, agent_type=types[i]))

for agent in agents:
    agent.neighbors = [agents[n] for n in G_full.neighbors(agent.id)]

model = LuxMarchesiModel(state, params, agents)
history = model.simulate(totalT, 1)

# Save the data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./files/tmp/simul{timestamp}.pkl"
with open(filename, "wb") as f:
    pickle.dump(history, f)
