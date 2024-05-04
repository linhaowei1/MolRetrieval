import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
from utils.env_utils import *

def init_chemberta():
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_PATH)
    model = AutoModel.from_pretrained(CHEMBERTA_PATH)
    model.cuda()
    model.eval()
    return tokenizer, model

@torch.no_grad()
def chemberta_embedding(smiles_block, tokenizer, model):
    batch = tokenizer(smiles_block, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    batch = {k: v.cuda() for k, v in batch.items()}
    embeddings = model(**batch)['last_hidden_state'][:, 0].detach().cpu()
    torch.cuda.empty_cache()
    return embeddings