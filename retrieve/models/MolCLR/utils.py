import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .dataset import MoleculeDataset, ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST
from .model import GINet

from utils.env_utils import *


embedType = 'MolCLR'
batch_size = 1000
chunksize = 1e6

def init_molclr():
    model = GINet(
        num_layer=5, 
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0, 
        pool='mean',
    )
    model.load_state_dict(torch.load(MOLCLR_PATH, map_location='cpu'))
    model.eval()
    
    model = model.cpu()
    return None, model

@torch.no_grad()
def molclr_embedding(smiles, place_holder, model):
    
    if isinstance(smiles, list):
        raise NotImplementedError

    mol = Chem.MolFromSmiles(smiles)
    
    type_idx = []
    chirality_idx = []
    atomic_number = []

    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum() if atom.GetAtomicNum() != 0 else 1))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return model(data)
    
    
@torch.no_grad()
def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    dataset = MoleculeDataset(SMILES_LIB_PATH)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=25, drop_last=False, shuffle=False)
    model = GINet(
        num_layer=config['model']['num_layer'], 
        emb_dim=config['model']['emb_dim'],
        feat_dim=config['model']['feat_dim'],
        drop_ratio=config['model']['drop_ratio'], 
        pool=config['model']['pool'],
    )
    model.load_state_dict(torch.load(MOLCLR_PATH, map_location='cpu'))
    model.eval()
    model = model.cuda()

    os.makedirs(os.path.join(EMBEDDING_DIR, embedType), exist_ok=True)

    embedding_list = []
    
    dataloader = iter(dataloader)
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size):
        data = next(dataloader)
        data = data.cuda()
        embedding_block = model(data)

        embedding_list.append(embedding_block.cpu())
        
        if len(embedding_list) * batch_size == chunksize:
            final_embedding = torch.cat(embedding_list, dim=0)
            torch.save(final_embedding, os.path.join(EMBEDDING_DIR, embedType, str(int(i // chunksize)) + '.pt'))
            print("saved", str(int(i // chunksize)) + '.pt')
            embedding_list = []
        
    final_embedding = torch.cat(embedding_list, dim=0)
    torch.save(final_embedding, os.path.join(EMBEDDING_DIR, embedType, str(int(i // chunksize)) + '.pt'))

if __name__ == "__main__":
    main()
