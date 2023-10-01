import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, HeteroData, DataLoader

import pdb

class HeOTADataset(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MD17` dataset 
        which is from `"Machine learning of accurate energy-conserving molecular force fields" <https://advances.sciencemag.org/content/3/5/e1603015.short>`_ paper. 
        MD17 is a collection of eight molecular dynamics simulations for small organic molecules. 
    
        Args:
            root (string): The dataset folder will be located at root/name.
            name (string): The name of dataset. Available dataset names are as follows: :obj:`aspirin`, :obj:`benzene_old`, :obj:`ethanol`, :obj:`malonaldehyde`, 
                :obj:`naphthalene`, :obj:`salicylic`, :obj:`toluene`, :obj:`uracil`. (default: :obj:`benzene_old`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = MD17(name='aspirin')
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(batch=[672], force=[672, 3], pos=[672, 3], ptr=[33], y=[32], z=[672])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The property (energy) for the graph (molecule).
        * :obj:`force`: The 3D force for atoms.
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs

    """
    def __init__(self, root = 'dataset/', name = 'ota1_dataset', transform = None, pre_transform = None, pre_filter = None):

        self.name = name
        self.folder = osp.join(root, self.name)

        super(HeOTADataset, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Cost_Map.npy', 'Guide.npy', 'Values.npy', 'PPConnection.npy', 'Placement.npy', 'PMConnection.npy', 'MMConnection.npy']

    @property
    def processed_file_names(self):
        return self.name + '_pyg.pt'

    def download(self):
        pass

    def process(self):
        # Cost Map
        data = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)

        # Guide
        data2 = np.load(osp.join(self.raw_dir, self.raw_file_names[1]), allow_pickle=True)

        # Values
        data3 = np.load(osp.join(self.raw_dir, self.raw_file_names[2]), allow_pickle=True)

        # PPConnection
        data4 = np.load(osp.join(self.raw_dir, self.raw_file_names[3]), allow_pickle=True)

        # Placement
        data5 = np.load(osp.join(self.raw_dir, self.raw_file_names[4]), allow_pickle=True)

        # PMConnection
        data6 = np.load(osp.join(self.raw_dir, self.raw_file_names[5]), allow_pickle=True)

        # MMConnection
        data7 = np.load(osp.join(self.raw_dir, self.raw_file_names[6]), allow_pickle=True)

        # label post_sim
        E = data3[:, :5]
        N = E.shape[0]
        E[:, 0] *= 1.0e+3
        E[:, 1] /= 1.0e+1
        E[:, 2] /= 1.0e+7
        E[:, 3] /= 1.0e+1
        E[:, 4] *= 1.0e+4

        # net idx
        Nid = data[:N, 0]

        # pin num
        Pnum = data[:N, 1]

        # pin idx
        Pid = data[:N, 2]

        # x, y, z position
        Pos = data[:N, 3]

        # PPConnection
        SRC_PP, TAG_PP = data4[:N, 0], data4[:N, 1]

        # PMConnection
        SRC_PM, TAG_PM = data6[:N, 0], data6[:N, 1]

        # MMConnection
        SRC_MM, TAG_MM = data7[:N, 0], data7[:N, 1]

        # module idx
        Midx = data5[:N, 0]
        # module type
        Mtype = data5[:N, 1]
        # module locations
        Mxl = data5[:N, 2]
        Myl = data5[:N, 3]
        Mxh = data5[:N, 4]
        Myh = data5[:N, 5]
        # module layers
        MLayers = data5[:N, 6]

        # Guide information
        G = np.asarray(data2[:N, :])

        data_list = []
        for i in tqdm(range(N)):
            N_pin = len(Nid[i])

            Nid_i  = torch.tensor(Nid[i], dtype=torch.int64).reshape(N_pin)
            Pnum_i = torch.tensor(Pnum[i], dtype=torch.int64).reshape(N_pin)
            Pid_i  = torch.tensor(Pid[i], dtype=torch.int64).reshape(N_pin)
            Pos_i  = torch.tensor(Pos[i], dtype=torch.float32).reshape(N_pin, 3)[:, [2, 1, 0]]

            Pos_i[:, :2] = Pos_i[:, :2]
            Pos_i[:, -1] = 200 * 3 * Pos_i[:, -1]   # grid size
            Pos_i /= 200

            # dummy labels
            E_i = torch.tensor(E[i], dtype=torch.float32)
            G_i = torch.tensor(G[i][-3:], dtype=torch.float32)
            G_i = G_i.repeat(N_pin, 1)
            NW_i = torch.tensor(G[i][:-3], dtype=torch.float32)

            data = HeteroData()
            data['ap'].num_nodes = N_pin
            data['ap'].nid = Nid_i
            data['ap'].pnum = Pnum_i
            data['ap'].pid = Pid_i
            data['ap'].pos = Pos_i[:, :]
            data['ap'].z = Pos_i[:, 0]

            # modules
            N_modules = len(Midx[i])
            data['module'].num_nodes = N_modules
            data['module'].midx = torch.tensor(Midx[i], dtype=torch.int64)
            data['module'].mtype = torch.tensor(Mtype[i], dtype=torch.int64)

            x = (torch.tensor(Mxl[i], dtype=torch.float32) + torch.tensor(Mxh[i], dtype=torch.float32)) / 2.0
            y = (torch.tensor(Myl[i], dtype=torch.float32) + torch.tensor(Myh[i], dtype=torch.float32)) / 2.0
            z = torch.mean(torch.tensor(MLayers[i], dtype=torch.float32), axis=1)
            data['module'].pos = torch.concat([x, y, z], axis=0).reshape(3, N_modules).transpose(0, 1)
            data['module'].pos /= 200

            # data['module'].mxl = torch.tensor(Mxl[i], dtype=torch.float32)
            # data['module'].myl = torch.tensor(Myl[i], dtype=torch.float32)
            # data['module'].mxh = torch.tensor(Mxh[i], dtype=torch.float32)
            # data['module'].myh = torch.tensor(Myh[i], dtype=torch.float32)

            data.g  = G_i
            data.nw = NW_i

            # Edges
            src_pp = torch.tensor(SRC_PP[i], dtype=torch.int64)
            tag_pp = torch.tensor(TAG_PP[i], dtype=torch.int64)

            c_edge_index = torch.vstack([src_pp, tag_pp])
            data['ap', 'connect', 'ap'].edge_index = c_edge_index

            src_pm = torch.tensor(SRC_PM[i], dtype=torch.int64)
            tag_pm = torch.tensor(TAG_PM[i], dtype=torch.int64)

            m_edge_index = torch.vstack([src_pm, tag_pm])
            data['ap', 'connect', 'module'].edge_index = m_edge_index

            src_mm = torch.tensor(SRC_MM[i], dtype=torch.int64)
            tag_mm = torch.tensor(TAG_MM[i], dtype=torch.int64)

            m_edge_index = torch.vstack([src_mm, tag_mm])
            data['module', 'connect', 'module'].edge_index = m_edge_index

            data.y = E_i
            data_list.append(data)

        pdb.set_trace()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict


if __name__ == '__main__':
    dataset = HeOTADataset(root = '/home/pxu/data/paroute/learning_datas/', name='heota1_medium')
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
    # print(data['module'].myl.shape)
    # print(data['module'].mxh.shape)
    # print(data['module'].myh.shape)

    N = len(dataset.data.y) // 5
    split_idx = dataset.get_idx_split(N, train_size=500, valid_size=100, seed=42)

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