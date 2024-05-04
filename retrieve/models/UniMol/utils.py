import torch
from tqdm import tqdm
import json
import selfies
from transformers import AutoTokenizer, AutoModel
from utils.env_utils import *
from unimol_tools import UniMolRepr
import logging

def init_UniMol():
    clf = UniMolRepr(data_type='molecule', remove_hs=True)
    return clf, 'placeholder'

@torch.no_grad()
def UniMol_embedding(smiles_block, clf, place_holder=None):
    if isinstance(smiles_block, str):
        smiles_block = [smiles_block]
    # TODO(haowei): unimol is very slow beacuse it has to optimize 3D conformation for every mol
    unimol_repr = clf.get_repr(smiles_block)
    return torch.tensor(unimol_repr['cls_repr'])