import torch
import itertools

from alpharoute.d3gnn.dataset import HeOTADataset
from alpharoute.d3gnn.method import HeSchNet #SchNet, DimeNetPP, ComENet
from alpharoute.d3gnn.method import run # main function
from alpharoute.d3gnn.evaluation import ThreeDEvaluator # MAE for QM9 and MD17

from alpharoute.relax import PA_Distribution, Potential, AnalogRelaxation

import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print(device)

# load dataset

dataset = HeOTADataset(root = '/home/pxu/data/paroute/DAC/', name='ota1_large')
data = dataset[0]

# print access point information
print(data['ap'])
print(data['ap'].nid.shape)
print(data['ap'].pnum.shape)
print(data['ap'].pid.shape)
print(data['ap'].pos.shape)
print(data['ap'].z.shape)

# print module information
print(data['module'])
print(data['module'].midx.shape)
print(data['module'].mtype.shape)
print(data['module'].pos.shape)

N = len(dataset.data.y) // 5
split_idx = dataset.get_idx_split(N, train_size=800, valid_size=3, seed=42)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

from torch_geometric.data import DataLoader

def mean_std_small(loader):
    data = next(iter(loader))
    y = data.y.reshape(-1, 5)
    mean, std = y.mean(dim=0), y.std(dim=0)
    return mean, std

train_loader = DataLoader(train_dataset, len(train_dataset), shuffle=True)
valid_loader = DataLoader(valid_dataset, len(valid_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

train_mean, train_std = mean_std_small(train_loader)
print("Train mean and std: ", train_mean, train_std)
val_mean, val_std = mean_std_small(valid_loader)
print("Val mean and std: ", val_mean, val_std)
test_mean, test_std = mean_std_small(test_loader)
print("Test mean and std: ", test_mean, test_std)

train_mean, train_std = train_mean.to(device), train_std.to(device)
val_mean, val_std = val_mean.to(device), val_std.to(device)

# Loading model, loss function, and evaluation function
model = HeSchNet(n_num=18, n_module=25, energy_and_force=False, cutoff=200, num_layers=6,
        hidden_channels=128, out_channels=5, num_filters=128, num_gaussians=500
)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Training
run3d = run()
run3d.run(device, train_dataset, valid_dataset, valid_dataset,
        model, loss_func, evaluation, 
        epochs=1, batch_size=1, vt_batch_size=1, lr=0.005, lr_decay_factor=0.5, lr_decay_step_size=15)

num_samples = 2
total_pa_lenth = 3

def get_pa_guide(num_samples, total_pa_lenth):
    params = []
    count = 1

    for i in range(total_pa_lenth):
        param = [0, 2, 4, 6, 8]
        params.append(param)
    return params

# hard code here
N_pin = 156
pa_guide_choices = get_pa_guide(num_samples, total_pa_lenth)
cost_guide_distribution = PA_Distribution(pa_guide_choices, N_pin)

model = model.to(device)

potential_function = Potential(model, valid_dataset)

# relaxer = AnalogRelaxation(cost_guide_distribution, potential_function, pool_size=10, max_iterations=1, max_outer_iterations=5)
import time

start_time = time.time()
relaxer = AnalogRelaxation(cost_guide_distribution, potential_function, pool_size=15, max_iterations=5, max_outer_iterations=3)
cost_guides, potentials = relaxer.process()
end_time = time.time()

print(f"Relaxation Time : {end_time - start_time}")

# with open("cost_guide_result.txt", "w") as file:
#     file.write(str(cost_guides))
#     file.write(str(potentials))