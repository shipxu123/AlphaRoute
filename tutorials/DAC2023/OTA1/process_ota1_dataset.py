import torch
from alpharoute.d3gnn.dataset import HeOTADataset
from torch_geometric.data import DataLoader

if __name__ == '__main__':
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
    split_idx = dataset.get_idx_split(N, train_size=500, valid_size=100, seed=42)

    import pdb

    print(dataset[split_idx['train']])
    print(dataset[split_idx['valid']])
    print(dataset[split_idx['test']])

    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    pdb.set_trace()

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    data = next(iter(train_loader))
    print(data.g)
    print(data.nw)
    pdb.set_trace()