from itertools import cycle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class ThompsonSamplingBase:
    def __init__(self, contents_ctr):
        """
        contents_ctr : [[0.1, 0.3, 0.3,0.1, ],[0.55, 0.9, 0.7],[0.8, 0.3, 0.4, 0.1]]
        """
        self.length = sum(len(ctr) for ctr in contents_ctr)
        self.gt = np.concatenate(contents_ctr)
        self.split(contents_ctr)
        self.regrets = []

    def split(self, contents_ctr):
        contents_ctr = [np.array(c) for c in contents_ctr]
        self.contents_ctr = contents_ctr[0]
        self.norm_contents_ctr = self.normalize(self.contents_ctr)
        self.cans_ctr = contents_ctr[1:]

    def initalize(self):
        self.candidates = [*range(len(self.contents_ctr))]
        self.time_decays = {i:1 for i in range(self.length)}
        self.cluster_idx = [len(self.contents_ctr)]
        self.selected = np.zeros(len(self.candidates))

    def normalize(self, ctr):
        return ctr / sum(ctr)

    def cluster_decay(self):
        pass

    def time_decay(self):
        for i in range(len(self.contents_ctr)):
            self.time_decays[i] += 1

    def update_arms(self, cluster_decay_size, topk):
        pass

    def generate_random_click(self):
        click = np.random.choice(self.candidates, p = self.norm_contents_ctr)
        
        return click

    def add_new_cluster(self, candidates):
        self.contents_ctr = np.concatenate([self.contents_ctr, candidates])
        self.selected = np.concatenate([self.selected, np.zeros_like(candidates)])
        self.candidates = list(range(len(self.contents_ctr)))
        self.norm_contents_ctr = self.normalize(self.contents_ctr)
        self.cluster_idx.append(len(self.contents_ctr))

    def update_regret(self, selected_arms):
        maximum_reward = self.norm_contents_ctr[selected_arms].max()
        regret = self.norm_contents_ctr.max() - maximum_reward

        # selected_sum = self.norm_contents_ctr[selected_arms].sum()
        # topk = len(selected_arms)
        # topk_sum = np.partiton(self.norm_contents_ctr, -topk)[-topk:].sum()
        # regret = topk_sum - selected_sum
        
        self.regrets.append(regret)

    def run(self, iterations = 100000, verbose = False, cumulative = True, cluster_decay_size = np.exp(1), topk = 3):
        self.initalize()

        grid_size = len(self.cans_ctr) + 1
        x = 0
        ploted = 0
        sec = int(iterations/grid_size)
        circle = cycle(range(grid_size))

        if verbose:
            fig, ax = plt.subplots(grid_size, grid_size, figsize = (8,6))

        for i in range(iterations):

            recommend = self.update_arms(cluster_decay_size = cluster_decay_size, topk = topk)
            self.update_regret(recommend)

            if i % (sec//grid_size) == 0:
                y = next(circle)

                if verbose:
                    ax[x,y].bar(self.candidates, self.selected)

                    if x == 0:
                        ax[x,y].set_title(f"{i}'th iter")
                    else:
                        ax[x,y].set_title(f"{x+1}'th cans added {i}'th iter")

                ploted += 1
                if ploted * (x+1) == grid_size**2:
                    break

                if ploted == grid_size:
                    x += 1
                    ploted = 0

                    candidates = self.cans_ctr.pop(0)
                    self.add_new_cluster(candidates)
                    
                    if not cumulative:
                        self.selected = np.zeros_like(self.selected)
            
        if verbose:
            print(self.norm_contents_ctr)
            print(self.selected/sum(self.selected))
            plt.show()

class DirichletThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr):
        super().__init__(contents_ctr)
        self.prior = np.ones_like(self.contents_ctr)
    
    def __str__(self):
        return 'Dirichlet'

    def cluster_decay(self, cluster_decay_size = np.exp(1)):
        updated_prior = np.copy(self.prior)
        curr_cluster = 0
        cluster_decay_factor = pow(cluster_decay_size, len(self.cluster_idx))

        for item in range(len(updated_prior)):

            if item >= self.cluster_idx[curr_cluster]:
                cluster_decay_factor /= cluster_decay_size
                curr_cluster += 1

            updated_prior[item] /= self.time_decays[item] * cluster_decay_factor
        return updated_prior

    def draw(self, prior, topk):
        rewards = np.random.dirichlet(prior, 1)
        recommend = rewards[0].argsort()[::-1]

        return recommend[:topk]

    def update_arms(self, cluster_decay_size = np.exp(1), topk = 3):
        updated_prior = self.cluster_decay(cluster_decay_size)

        recommend = self.draw(updated_prior, topk)
        click = self.generate_random_click()

        for i in recommend:
            self.selected[i] += 1
        self.prior[click] += 1

        self.time_decay()

        return recommend

    def add_new_cluster(self, candidates):
        super().add_new_cluster(candidates)
        self.prior = np.concatenate([self.prior, np.ones_like(candidates)])


class BetaThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr):
        super().__init__(contents_ctr)
        self.alpha, self.beta = np.ones_like(self.contents_ctr), np.ones_like(self.contents_ctr)
    
    def __str__(self):
        return 'Beta'

    def cluster_decay(self, cluster_decay_size = np.exp(1)):
        updated_alpha = np.copy(self.alpha)
        updated_beta = np.copy(self.beta)

        curr_cluster = 0
        cluster_decay_factor = pow(cluster_decay_size, len(self.cluster_idx))

        for item in range(len(updated_alpha)):

            if item >= self.cluster_idx[curr_cluster]:
                cluster_decay_factor /= cluster_decay_size
                curr_cluster += 1

            updated_alpha[item] /= self.time_decays[item] * cluster_decay_factor
            updated_beta[item] /= self.time_decays[item] * cluster_decay_factor

        return updated_alpha, updated_beta

    def draw(self, alpha, beta, topk):
        rewards = np.random.beta(alpha, beta)
        recommend = rewards.argsort()[::-1]

        return recommend[:topk]

    def update_arms(self, cluster_decay_size = np.exp(1), topk = 3):
        updated_alpha, updated_beta = self.cluster_decay(cluster_decay_size)

        recommend = self.draw(updated_alpha, updated_beta, topk)
        click = self.generate_random_click()

        for item in recommend:
            self.selected[item] += 1

            if item == click:
                self.alpha[item] += 1
            else:
                self.beta[item] += 1

        self.time_decay()

        return recommend

    def add_new_cluster(self, candidates):
        super().add_new_cluster(candidates)
        self.alpha = np.concatenate([self.alpha, np.ones_like(candidates)])
        self.beta = np.concatenate([self.beta, np.ones_like(candidates)])

def beta_dist(alpha, beta, ground_truth):

    x = np.linspace(0, 1, 1002)[1:-1]
    plt.figure(figsize=(10,5))

    for i, (a,b,gt) in enumerate(zip(alpha, beta, ground_truth)):
        dist = stats.beta(a,b)
        dist_y = dist.pdf(x)
        plt.plot(x,dist_y,label = f'{i}th : ( {a} , {b} ) / E : {a/(a+b):.3f} / R : {gt:.3f}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    dts = DirichletThompsonSampling(contents_ctr =
                        [
                        [0.1, 0.3, 0.3,0.1,],
                        [0.55, 0.9, 0.7],
                        [0.8, 0.3, 0.4, 0.1]
                        ])
    dts.run(iterations = 100000, verbose = True, cluster_decay_size = np.exp(1))

    bts = BetaThompsonSampling(contents_ctr =
                        [
                        [0.1, 0.3, 0.3,0.1,],
                        [0.55, 0.9, 0.7],
                        [0.8, 0.3, 0.4, 0.1]
                        ])

    bts.run(iterations = 100000, verbose = True, cluster_decay_size = np.exp(1))
