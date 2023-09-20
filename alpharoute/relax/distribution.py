import torch
from torch.distributions.categorical import Categorical

class PA_Distribution(object):
    def __init__(self, cost_guide_choice, N_pin):
        tensor_cost_guide_choice = torch.Tensor(cost_guide_choice)
        # min_choice = torch.max(tensor_cost_guide_choice).int()
        max_cost = torch.max(tensor_cost_guide_choice).int()
        tensor_cost_guide = torch.zeros([len(cost_guide_choice), max_cost + 1])

        for i, c in enumerate(cost_guide_choice):
            tensor_cost_guide[i, c] += 1.0 / len(c)

        self.distribution = Categorical(tensor_cost_guide)
        self.N_pin = N_pin

    def sample(self, size):
        cost_guides = [[self.distribution.sample().tolist()] * self.N_pin for i in range(size)]
        return cost_guides

    def add_noise(self, cost_guide):
        # equals to resample
        new_cost_guide = self.distribution.sample().tolist()
        new_cost_guide = [new_cost_guide] * self.N_pin
        return new_cost_guide