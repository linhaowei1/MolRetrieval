import yaml
import argparse
import os
from rdkit import Chem
import time

from retrieve.utils import Retriever
from utils.env_utils import *

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='config.yaml')
    return args.parse_args()

def init_retriever(config):
    retriever = Retriever(
        embedding_type = config['embedding_type'],
        distance_type = config['distance_type'],
        embedding_dir = EMBEDDING_DIR,
        prunning = config['prunning'],
        smiles_lib_path = SMILES_LIB_PATH,
        block_size = BLOCKSIZE
    )
    return retriever


if __name__ == '__main__':
    args = get_args()
    
    retrieve_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    retriever = init_retriever(retrieve_config)
    
    queries = retriever.recieve_query(retrieve_config['query_path'])
    
    begin = time.time()
    smiles_list_of_list = retriever(queries, topk=retrieve_config['topk'])
    end = time.time()
    
    retrieve_config['retrieve_time'] = end - begin
    
    # TODO(haowei): this is hard coded...
    save_path = os.path.join(retrieve_config['save_path'], 
                             f'{retrieve_config["embedding_type"]}_{retrieve_config["distance_type"]}_top{retrieve_config["topk"]}')
    
    if retrieve_config['prunning']:
        save_path += '_prunning'
    
    os.makedirs(save_path, exist_ok=True)
    
    retriever.save_result(smiles_list_of_list, save_path)
    
    yaml.safe_dump(retrieve_config, open(os.path.join(save_path, 'config.yaml'), 'w'))