
from retrieve.utils import *
from utils.env_utils import *
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_type', type=str, choices=['MACCSkeys', 'RDKFingerprint', 'EStateFingerprint', 'ChemBERTa', 'MolT5', 'BioT5'], required=True)
args = parser.parse_args()


blocksize = 5000
chunksize = 1e6
assert chunksize % blocksize == 0, "chunksize must be divisible by blocksize"

with open(SMILES_LIB_PATH, 'r') as f:
    smiles = [x.strip().split()[0] for x in f.readlines()]

embedType = args.embedding_type

print("\n#### embedding:", embedType)

embedding = Embedding(embedType)

embedding_list = []

os.makedirs(os.path.join(EMBEDDING_DIR, embedType), exist_ok=True)

for i in tqdm(range(0, len(smiles), blocksize), total=len(smiles)//blocksize):
    
    if args.embedding_type in ['ChemBERTa', 'MolT5', 'BioT5']:
        smiles_block = smiles[i:i+blocksize]
    else:
        smiles_block = [Chem.MolFromSmiles(x) for x in smiles[i:i+blocksize]]
    
    embedding_block = embedding(smiles_block)
    embedding_list.append(embedding_block)
    if len(embedding_list) * blocksize == chunksize:
        final_embedding = torch.cat(embedding_list, dim=0)
    
        Embedding.save(final_embedding, os.path.join(EMBEDDING_DIR, embedType, str(int(i // chunksize)) + '.pt'))
        print("saved", str(int(i // chunksize)) + '.pt')
        embedding_list = []
        
final_embedding = torch.cat(embedding_list, dim=0)
Embedding.save(final_embedding, os.path.join(EMBEDDING_DIR, embedType, str(int(i // chunksize)) + '.pt'))
