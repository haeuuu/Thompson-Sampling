import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy import stats

from thompson_sampling import ThompsonSamplingBase


class ThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr):
        super().__init__(contents_ctr)
        self.alpha, self.beta = np.ones_like(self.contents_ctr), np.ones_like(self.contents_ctr)

    def __str__(self):
        return 'vanilla TS'

    def draw(self, topk):
        rewards = np.random.beta(self.alpha, self.beta)
        recommend = rewards.argsort()[::-1]

        return recommend[:topk]

    def update_arms(self, cluster_decay_size = None, topk = 3):
        recommend = self.draw(topk)
        click = self.generate_random_click()

        for item in recommend:
            self.selected[item] += 1

            if item == click:
                self.alpha[item] += 1
            else:
                self.beta[item] += 1

        return recommend

    def add_new_cluster(self, candidates):
        super().add_new_cluster(candidates)
        self.alpha = np.concatenate([self.alpha, np.ones_like(candidates)])
        self.beta = np.concatenate([self.beta, np.ones_like(candidates)])

class DiscountedThompsonSampling(ThompsonSamplingBase):
    def __init__(self, contents_ctr, alpha_init = 0.5, beta_init = 0.5, gamma = 0.9):
        super().__init__(contents_ctr)
        self.alpha, self.beta = np.zeros_like(self.contents_ctr), np.zeros_like(self.contents_ctr)
        self.alpha_init, self.beta_init = alpha_init, beta_init
        self.gamma = gamma

    def __str__(self):
        return 'discounted TS'

    def discount(self):
        self.alpha *= self.gamma
        self.beta *= self.gamma

    def draw(self, topk):
        rewards = np.random.beta(self.alpha + self.alpha_init, self.beta + self.beta_init)
        recommend = rewards.argsort()[::-1]

        return recommend[:topk]

    def update_arms(self, cluster_decay_size = None, topk = 3):
        
        recommend = self.draw(topk)
        click = self.generate_random_click()
        
        self.discount()
        
        for item in recommend:
            self.selected[item] += 1

            if item == click:
                self.alpha[item] += 1
            else:
                self.beta[item] += 1

        return recommend

    def add_new_cluster(self, candidates):
        super().add_new_cluster(candidates)
        self.alpha = np.concatenate([self.alpha, np.zeros_like(candidates)])
        self.beta = np.concatenate([self.beta, np.zeros_like(candidates)])

class DirichletDiscountedTS(ThompsonSamplingBase):
    def __init__(self, contents_ctr, prior_init = 0.5, gamma = 0.9):
        super().__init__(contents_ctr)
        self.prior = np.zeros_like(self.contents_ctr)
        self.prior_init = 1
        self.gamma = gamma

    def __str__(self):
        return 'Dirichlet discounted TS'

    def discount(self):
        self.prior *= self.gamma

    def draw(self, topk):
        rewards = np.random.dirichlet(self.prior + self.prior_init, 1)
        recommend = rewards[0].argsort()[::-1]

        return recommend[:topk]

    def update_arms(self, cluster_decay_size = None, topk = 3):
        
        recommend = self.draw(topk)
        click = self.generate_random_click()

        self.discount()

        for i in recommend:
            self.selected[i] += 1
        self.prior[click] += 1

        return recommend

    def add_new_cluster(self, candidates):
        super().add_new_cluster(candidates)
        self.prior = np.concatenate([self.prior, np.zeros_like(candidates)])

class DiscountedOptimisticThompsonSampling(DiscountedThompsonSampling):
    def __init__(self, contents_ctr, alpha_init = 0.5, beta_init = 0.5, gamma = 0.9):
        super().__init__(contents_ctr, alpha_init, beta_init, gamma)

    def __str__(self):
        return 'dicounted Optimistic TS'

    def draw(self, topk):
        rewards = np.random.beta(self.alpha + self.alpha_init, self.beta + self.beta_init)
        mean = self.alpha / (self.alpha + self.beta)
        recommend = np.maximum(rewards, mean).argsort()[::-1]

        return recommend[:topk]