import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy import stats


class ThompsonSamplingBase:
    def __init__(self, contents_ctr):
        """
        contents_ctr : [[0.1, 0.3, 0.3,0.1, ],[0.55, 0.9, 0.7],[0.8, 0.3, 0.4, 0.1]]
        """
        self.length = sum(len(ctr) for ctr in contents_ctr)
        self.gt = np.concatenate(contents_ctr)
        self.split(contents_ctr)

    def split(self, contents_ctr):
        contents_ctr = [np.array(c) for c in contents_ctr]
        self.contents_ctr = contents_ctr[0]
        self.norm_contents_ctr = self.normalize(self.contents_ctr)
        self.cans_ctr = contents_ctr[1:]

    def initalize(self):
        self.candidates = [*range(len(self.contents_ctr))]
        self.time_decays = {i:1 for i in range(self.length)}
        self.cluster_idx = [len(self.contents_ctr)]
        self.selected = [0]*len(self.candidates)

    def normalize(self, ctr):
        return ctr / sum(ctr)

    def cluster_decay(self):
        pass

    def time_decay(self):
        for i in range(len(self.contents_ctr)):
            self.time_decays[i] += 1

    def updated_arms(self, i):
        pass

    def add_new_cluster(self):
        pass

    def run(self, iteration = 100):
        self.initalize()

        grid_size = 3
        x = 0
        ploted = 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize = (8,6))
        sec = int(iteration/grid_size)
        circle = cycle(range(grid_size))

        for i in range(iteration):

            self.updated_arms(i)

            if i % (sec//grid_size) == 0:
                y = next(circle)
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
                    self.add_new_cluster()

        print(self.norm_contents_ctr)
        selected = np.array(self.selected)
        print(selected/sum(selected))
        plt.show()

class DirichletThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr):
        super().__init__(contents_ctr)
        self.prior = np.ones_like(self.contents_ctr)

    def cluster_decay(self, updated_prior):
        curr_cluster = 0
        cluster_decay_factor= pow(10., len(self.cluster_idx))

        for item in range(len(updated_prior)):

            if item >= self.cluster_idx[curr_cluster]:
                cluster_decay_factor /= 10.
                curr_cluster += 1

            updated_prior[item] /= self.time_decays[item] * cluster_decay_factor

    def draw(self, prior):
        recommend = np.random.dirichlet(prior, 1)
        res = recommend[0].argsort()[::-1]

        return res

    def updated_arms(self, iteration, topk = 3):
        updated_prior = np.copy(self.prior)

        self.cluster_decay(updated_prior)

        res = self.draw(updated_prior)
        for k in range(topk):
            self.selected[res[k]] += 1

        click = np.random.choice(self.candidates, p = self.norm_contents_ctr)
        self.prior[click] += 1

        self.time_decay()

        if iteration%10000 == 0:
            print('topk : ', res)

    def add_new_cluster(self):
        candidates = self.cans_ctr.pop(0)
        self.contents_ctr = np.concatenate([self.contents_ctr, candidates])
        self.selected = np.concatenate([self.selected, np.zeros_like(candidates)])
        self.candidates = list(range(len(self.contents_ctr)))
        self.norm_contents_ctr = self.normalize(self.contents_ctr)
        self.cluster_idx.append(len(self.contents_ctr))

        self.prior = np.concatenate([self.prior, np.ones_like(candidates)])


class BetaThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr):
        super().__init__(contents_ctr)
        self.alpha, self.beta = np.ones_like(self.contents_ctr), np.ones_like(self.contents_ctr)

    def cluster_decay(self, updated_alpha, updated_beta):
        curr_cluster = 0
        cluster_decay_factor = pow(10., len(self.cluster_idx))

        for item in range(len(updated_alpha)):

            if item >= self.cluster_idx[curr_cluster]:
                cluster_decay_factor /= 10.
                curr_cluster += 1

            updated_alpha[item] /= self.time_decays[item] * cluster_decay_factor
            updated_beta[item] /= self.time_decays[item] * cluster_decay_factor

    def draw(self, alpha, beta):
        score = []
        for a, b in zip(alpha, beta):
            reward = np.random.beta(a, b)
            score.append(reward)

        return np.array(score).argsort()[::-1]

    def updated_arms(self, iteration, topk=3):
        updated_alpha = np.copy(self.alpha)
        updated_beta = np.copy(self.beta)
        self.cluster_decay(updated_alpha, updated_beta)

        res = self.draw(updated_alpha, updated_beta)
        click = np.random.choice(self.candidates, p=self.norm_contents_ctr)

        for item in res[:topk]:
            self.selected[item] += 1

            if item == click:
                self.alpha[item] += 1
            else:
                self.beta[item] += 1

        self.time_decay()

        if iteration % 10000 == 0:
            print('topk : ', res)

            # x = np.linspace(0, 1, 1002)[1:-1]
            # plt.figure(figsize=(10,5))
            #
            # for i,(a,b) in enumerate(zip(self.alpha, self.beta)):
            #     dist = stats.beta(a,b)
            #     dist_y = dist.pdf(x)
            #     plt.plot(x,dist_y,label = f'{i}th : ( {a} , {b} ) / E : {a/(a+b):.3f} / R : {self.norm_contents_ctr[i]:.3f}')
            #
            # plt.legend()
            # plt.show()

    def add_new_cluster(self):
        candidates = self.cans_ctr.pop(0)
        self.contents_ctr = np.concatenate([self.contents_ctr, candidates])
        self.selected = np.concatenate([self.selected, np.zeros_like(candidates)])
        self.candidates = list(range(len(self.contents_ctr)))
        self.norm_contents_ctr = self.normalize(self.contents_ctr)
        self.cluster_idx.append(len(self.contents_ctr))

        self.alpha = np.concatenate([self.alpha, np.ones_like(candidates)])
        self.beta = np.concatenate([self.beta, np.ones_like(candidates)])

def beta_dist(bts):

    x = np.linspace(0, 1, 1002)[1:-1]
    plt.figure(figsize=(10,5))

    for i,(a,b) in enumerate(zip(bts.alpha, bts.beta)):
        dist = stats.beta(a,b)
        dist_y = dist.pdf(x)
        plt.plot(x,dist_y,label = f'{i}th : ( {a} , {b} ) / E : {a/(a+b):.3f} / R : {bts.norm_contents_ctr[i]:.3f}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    dts = DirichletThompsonSampling(contents_ctr =
                        [
                        [0.1, 0.3, 0.3,0.1,],
                        [0.55, 0.9, 0.7],
                        [0.8, 0.3, 0.4, 0.1]
                        ])
    dts.run(100000)

    bts = BetaThompsonSampling(contents_ctr =
                        [
                        [0.1, 0.3, 0.3,0.1,],
                        [0.55, 0.9, 0.7],
                        [0.8, 0.3, 0.4, 0.1]
                        ])
    bts.run(100000)
