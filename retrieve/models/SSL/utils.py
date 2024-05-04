import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from rdkit import Chem

from .dataset import MoleculeDataset, mol_to_graph_data_obj_simple
from .model import GNN
from .graphMVP import GNNComplete

from tqdm import tqdm
from utils.env_utils import *

embedType = {
    'GROVER': MOTIF_PATH,
    'AttrMask': ATTRMASK_PATH,
    'GPT-GNN': GPTGNN_PATH,
    'GraphCL': GRAPHCL_PATH,
    'GraphMVP': GRAPHMVP_PATH
}

batch_size = 1000
chunksize = 1e6

def init_ssl(emb_type):
    if emb_type == 'GraphMVP':
        model = GNNComplete(num_layer=5, emb_dim=300)
    else:
        model = GNN(num_layer=5, emb_dim=300)
    model.load_state_dict(torch.load(embedType[emb_type], map_location='cpu'))
    model.eval()
    return None, model

def ssl_embedding(smiles, placeholder, model):
    if isinstance(smiles, list):
        raise NotImplementedError
    
    mol = Chem.MolFromSmiles(smiles)
    
    data = mol_to_graph_data_obj_simple(mol)
    return model(data)


@torch.no_grad()
def main(emb_type):

    dataset = MoleculeDataset(SMILES_LIB_PATH)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=25, drop_last=False, shuffle=False)
    if emb_type == 'GraphMVP':
        model = GNNComplete(num_layer=5, emb_dim=300)
    else:
        model = GNN(num_layer=5, emb_dim=300)
    model.load_state_dict(torch.load(embedType[emb_type], map_location='cpu'))
    model.eval()
    model = model.cuda()

    os.makedirs(os.path.join(EMBEDDING_DIR, emb_type), exist_ok=True)

    embedding_list = []
    
    dataloader = iter(dataloader)
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size):
        data = next(dataloader)
        data = data.cuda()
        embedding_block = model(data)
        embedding_list.append(embedding_block.cpu())
        
        if len(embedding_list) * batch_size == chunksize:
            final_embedding = torch.cat(embedding_list, dim=0)
            torch.save(final_embedding, os.path.join(EMBEDDING_DIR, emb_type, str(int(i // chunksize)) + '.pt'))
            print("saved", str(int(i // chunksize)) + '.pt')
            embedding_list = []
        
    final_embedding = torch.cat(embedding_list, dim=0)
    torch.save(final_embedding, os.path.join(EMBEDDING_DIR, emb_type, str(int(i // chunksize)) + '.pt'))

if __name__ == "__main__":
    for emb_type in embedType.keys():
        print(emb_type)
        main(emb_type)
