import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.data import DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

class Potential(object):
    def __init__(self, model, dataset, r=1.0e-10):
        self.model = model
        self.r = r
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()

    def forward(self, cost_guide, evaluate=True):
        if(evaluate):
            print(f"evaluate potential of {cost_guide.int().tolist()}")
        else:
            print(f"optimize potential of {cost_guide.int().tolist()}")

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)
        scale = torch.Tensor([-1.0, 1.0, 1.0, 1.0, -1.0]).to(device)
        loss_accum = 0

        for step, data in enumerate(self.data_loader):
            data = data.to(device)
            assert(data.g.shape == cost_guide.shape)
            data.g = cost_guide
            out = self.model(data)

            loss = torch.sum(out * scale) + self.r * torch.sum(torch.log(data.g + 1)) + self.r * torch.sum(torch.log(200 - data.g))
            if evaluate:
                loss_accum += loss.detach().cpu()
            else:
                loss_accum += loss

        return loss_accum / (step + 1)


    def optimize(self, cost_guide, max_iterations, alpha=0.01, epsilon=0.001):
        X = torch.Tensor(cost_guide).to(device).requires_grad_()

        optimizer = torch.optim.LBFGS([X], lr=0.01)

        for step in tqdm(range(max_iterations)):
            def closure():
                optimizer.zero_grad()
                loss = self.forward(X, evaluate=False)
                loss.backward()
                return loss
            optimizer.step(closure)

        return X.int().tolist(), self.forward(torch.floor(X)).item()
        # X = X.requires_grad_()

        # out = self.forward(X, evaluate=False)
        # loss = torch.sum(out)
        # loss.backward()

        # new_X = X + alpha * X.grad.sign()
        # X = torch.clamp(new_X, min=0, max=10).detach_()