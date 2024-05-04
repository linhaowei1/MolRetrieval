import torch
from tqdm import tqdm
import json
import selfies
from transformers import AutoTokenizer, AutoModel
from utils.env_utils import *

def init_biot5():
    tokenizer = AutoTokenizer.from_pretrained(BIOT5_PATH)
    model = AutoModel.from_pretrained(BIOT5_PATH)
    model.cuda()
    model.eval()
    return tokenizer, model

def smiles_to_selfies(smiles): 
    try:
        sel = selfies.encoder(smiles)
    except:
        sel = '[C]'
    return sel
        
@torch.no_grad()
def biot5_embedding(smiles_block, tokenizer, model):
    if isinstance(smiles_block, str):
        smiles_block = [smiles_block]
    smiles_block = [smiles_to_selfies(smiles) for smiles in smiles_block]
    batch = tokenizer(smiles_block, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    batch = {k: v.cuda() for k, v in batch.items()}
    embeddings = model.encoder(**batch)['last_hidden_state'][:, -1].detach().cpu()
    torch.cuda.empty_cache()
    return embeddings