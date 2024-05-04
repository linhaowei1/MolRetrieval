import torch
from tqdm import tqdm
import json
import selfies
from transformers import AutoTokenizer, AutoModel
from utils.env_utils import *

def init_molt5():
    tokenizer = AutoTokenizer.from_pretrained(MOLT5_PATH)
    model = AutoModel.from_pretrained(MOLT5_PATH)
    model.cuda()
    model.eval()
    return tokenizer, model

@torch.no_grad()
def molt5_embedding(smiles_block, tokenizer, model):
    batch = tokenizer(smiles_block, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    batch = {k: v.cuda() for k, v in batch.items()}
    embeddings = model.encoder(**batch)['last_hidden_state'][:, -1].detach().cpu()
    torch.cuda.empty_cache()
    return embeddings