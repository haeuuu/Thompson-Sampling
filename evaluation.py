from tqdm import tqdm
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from thompson_sampling import *

def evaluate(model, contents_ctr, experiments, **params):
    model_parameters, mab_parameters = {}, {}
    for key in list(params.keys()):
        if key.startswith('_'):
            model_parameters[key[1:]] = params.pop(key)
        else:
            mab_parameters[key] = params.pop(key)

    _regrets = []
    for e in tqdm(range(experiments)):
        m = model(contents_ctr, **model_parameters)
        m.run(**mab_parameters)
        _regrets.append(m.regrets)

    regrets_per_iter = np.mean(_regrets, axis = 0)

    return regrets_per_iter

def barplot(contents_ctr):
    base_colors = cycle(mcolors.BASE_COLORS.values())

    flatten = []
    colors = []
    for ctr in contents_ctr:
        color = next(base_colors)
        if color == (1,1,1): # black
            color = next(base_colors)

        flatten.extend(ctr)
        colors.extend([color]*len(ctr))

    plt.bar(range(len(flatten)), flatten, color = colors)
    plt.show()

def draw_regret(regrets, n_groups = 1000):
    s = {}
    for model, regrets in regrets.items():
        s[model] = np.array(list(map(lambda x: x.mean(), np.array_split(regrets, n_groups))))

    fig, ax = plt.subplots(figsize = (20, 10))
    for model, average in s.items():
        ax.plot(list(range(len(average))), average, label = model)
        # ax.plot(list(range(len(average))), average, 'o' ,markersize = 2, label = model)

    plt.legend(bbox_to_anchor=(0, 0))
    plt.show()

if __name__ == '__main__':
    contents_ctr = [[0.1, 0.3, 0.3,0.1],
                [0.55, 0.9, 0.7],
                [0.8, 0.3, 0.4, 0.1]]

    beta_10 = evaluate(BetaThompsonSampling,
                     contents_ctr,
                     experiments = 10,
                     cluster_decay_size = 10,
                     iterations = 100000,
                     topk = 3)

    dirichlet_10 = evaluate(DirichletThompsonSampling,
                     contents_ctr,
                     experiments = 10,
                     cluster_decay_size = 10,
                     iterations = 100000,
                     topk = 3)

    barplot(contents_ctr)
    draw_regret({'dirichlet_10' : dirichlet_10, 'beta_10' : beta_10},
           n_groups = 500)