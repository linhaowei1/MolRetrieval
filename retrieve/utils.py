from typing import List, Tuple, Dict, Union, Any, Optional
import os
import bisect
import logging
import torch
import heapq
import random
from functools import partial

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.EState import Fingerprinter

from .models.ChemBERTa.utils import *
from .models.BioT5.utils import *
from .models.MolT5.utils import *
from .models.SSL.utils import *
from .models.MolCLR.utils import *

similarityFunctions = {
    'Tanimoto': DataStructs.TanimotoSimilarity,
    'Dice': DataStructs.DiceSimilarity,
    'Cosine': DataStructs.CosineSimilarity,
    'Sokal': DataStructs.SokalSimilarity,
    'Russel': DataStructs.RusselSimilarity,
    'Kulczynski': DataStructs.KulczynskiSimilarity,
    'McConnaughey': DataStructs.McConnaugheySimilarity,
}

embeddingFunctions = {
    'RDKFingerprint': Chem.RDKFingerprint, 
    'MACCSkeys': MACCSkeys.GenMACCSKeys,
    'EStateFingerprint': Fingerprinter.FingerprintMol,
}

distanceType = (
    'Tanimoto',
    'Dice',
    'Cosine',
    'Euclidean',
    'Sokal',
    'Russel',
    'Kulczynski',
    'McConnaughey',
    'random'
)

embeddingTypes = (
    'RDKFingerprint', 
    'MACCSkeys',
    # 'AtomPairFingerprint', TODO(haowei): this is very slow
    # 'TopologicalTorsionFingerprint',  TODO(haowei): this is not working
    # 'MorganFingerprint', TODO(haowei): this is very slow
    'EStateFingerprint',
    'ChemBERTa',
    'MolT5',
    'BioT5',
    'UniMol',
    'AttrMask',
    'GPT-GNN',
    'GraphCL',
    'MolCLR',
    'GraphMVP',
    'GROVER',
    'random'
)

modelBasedEmbeddingTypes = {
    "ChemBERTa": [init_chemberta, chemberta_embedding], 
    "MolT5": [init_molt5, molt5_embedding], 
    "BioT5": [init_biot5, biot5_embedding],
    'MolCLR': [init_molclr, molclr_embedding],
    'GROVER': [partial(init_ssl, 'GROVER'), ssl_embedding],
    'AttrMask': [partial(init_ssl, 'AttrMask'), ssl_embedding],
    'GPT-GNN': [partial(init_ssl, 'GPT-GNN'), ssl_embedding],
    'GraphCL': [partial(init_ssl, 'GraphCL'), ssl_embedding],
    'GraphMVP': [partial(init_ssl, 'GraphMVP'), ssl_embedding],
}

class Embedding:
    
    def __init__(self, embedding_type: str):
        self.embedding_type = embedding_type
        assert embedding_type in embeddingTypes, "embedding type not implemented!"
        if embedding_type in modelBasedEmbeddingTypes:
            self.tokenizer, self.model = modelBasedEmbeddingTypes[embedding_type][0]()
            self.embedding_func = modelBasedEmbeddingTypes[embedding_type][1]
    
    def _embedding(self, mol: Chem.Mol) -> torch.Tensor:
        if self.embedding_type == 'RDKFingerprint':
            return torch.tensor(Chem.RDKFingerprint(mol)).float()
        elif self.embedding_type == 'MACCSkeys':
            return torch.tensor(MACCSkeys.GenMACCSKeys(mol)).float()
        elif self.embedding_type == 'AtomPairFingerprint':
            return torch.tensor(Pairs.GetAtomPairFingerprintAsBitVect(mol)).float()
        elif self.embedding_type == 'TopologicalTorsionFingerprint':
            return torch.tensor(Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol).ToList()).float()
        elif self.embedding_type == 'MorganFingerprint':
            return torch.tensor(AllChem.GetMorganFingerprint(mol, 2).ToList()).float()
        elif self.embedding_type == 'EStateFingerprint':
            return torch.tensor(Fingerprinter.FingerprintMol(mol)[0]).float()
            
    def __call__(self, mols: Union[List[Chem.Mol], Chem.Mol, List[str]]) -> torch.FloatTensor:
        '''
            mols: List of rdkit.Chem.Mol or rdkit.Chem.Mol
            return: List of torch.Tensor or torch.Tensor
        '''
        
        if self.embedding_type in modelBasedEmbeddingTypes:
            if isinstance(mols, list):
                batch_size = 5000    # FIXME(haowei): this is hard coded now.
                embeddings = []
                for i in range(0, len(mols), batch_size):
                    embedding = self.embedding_func(mols[i:i+batch_size], self.tokenizer, self.model)
                    if len(embedding.shape) == 1:
                        embedding = embedding.unsqueeze(0)
                    embeddings.append(embedding)
                return torch.cat(embeddings, dim=0)
            else:
                embedding = self.embedding_func(Chem.MolToSmiles(mols), self.tokenizer, self.model)
                return embedding
        else:
            if isinstance(mols, list):
                return torch.cat([self._embedding(mol).unsqueeze(0) for mol in mols], dim=0)
            elif isinstance(mols, Chem.Mol):
                return self._embedding(mols).unsqueeze(0)
            else:
                raise NotImplementedError("mols must be either a list of rdkit.Chem.Mol or rdkit.Chem.Mol")
    
    @classmethod
    def save(self, tensors: torch.FloatTensor, fn: str):
        torch.save(tensors, fn)
    
    @classmethod
    def load(self, fn: str) -> torch.FloatTensor:
        return torch.load(fn)
    
class Distance:
    
    def __init__(self, distance_type: str):
        self.distance_type = distance_type
        assert distance_type in distanceType, 'distance type not implemented!'
    
    def _Tanimoto(self, vecA: torch.Tensor, vecB: torch.Tensor):
        intersection = torch.mm(vecA, vecB.t())
        vecA_sum = torch.sum(vecA, dim=1, keepdim=True)
        vecB_sum = torch.sum(vecB, dim=1, keepdim=True)
        return intersection / (vecA_sum + vecB_sum - intersection)

    def _dice(self, vecA: torch.Tensor, vecB: torch.Tensor):
        intersection = torch.mm(vecA, vecB.t())
        vecA_sum = torch.sum(vecA, dim=1, keepdim=True)
        vecB_sum = torch.sum(vecB, dim=1, keepdim=True)
        return 2 * intersection / (vecA_sum + vecB_sum)
    
    def _cosine(self, vecA: torch.Tensor, vecB: torch.Tensor):
        vecA = vecA / torch.norm(vecA, dim=1, keepdim=True)
        vecB = vecB / torch.norm(vecB, dim=1, keepdim=True)
        return torch.mm(vecA, vecB.t())

    def _euclidean(self, vecA: torch.Tensor, vecB: torch.Tensor):
        vecA = vecA.unsqueeze(1).expand(vecA.shape[0], vecB.shape[0], vecA.shape[1])
        vecB = vecB.unsqueeze(0).expand(vecA.shape[0], vecB.shape[0], vecB.shape[1])
        return torch.norm(vecA - vecB, dim=2).squeeze()
    
    def _sokal(self, vecA: torch.Tensor, vecB: torch.Tensor):
        N11 = torch.mm(vecA, vecB.t())
        N10 = torch.mm(vecA, (1 - vecB).t())
        N01 = torch.mm(1 - vecA, vecB.t())    
        return N11 / (N11 + N10 * 2 + N01 * 2)

    def _russel(self, vecA: torch.Tensor, vecB: torch.Tensor):
        intersection = torch.mm(vecA, vecB.t())
        return intersection / vecA.shape[1]

    def _kulczynski(self, vecA: torch.Tensor, vecB: torch.Tensor):
        N11 = torch.mm(vecA, vecB.t())
        N10 = torch.mm(vecA, (1 - vecB).t())
        N01 = torch.mm(1 - vecA, vecB.t())
        return 0.5 * ((N11 / (N11 + N10)) + (N11 / (N11 + N01)))

    def _McConnaughey(self, vecA: torch.Tensor, vecB: torch.Tensor):
        N11 = torch.mm(vecA, vecB.t())
        N10 = torch.mm(vecA, (1 - vecB).t())
        N01 = torch.mm(1 - vecA, vecB.t())

        return (N11 * N11 - N10 * N01) / ((N11 + N10) * (N11 + N01))
    
    @torch.no_grad()
    def distance(self, vecA: torch.Tensor, vecB: torch.Tensor):
        '''
            vecA: torch.Tensor with shape (m,d)
            vecB: torch.Tensor with shape (n,d)
        '''
        assert vecB.shape[1] == vecA.shape[1], 'vecA and vecB must have the same dimension.'
        
        # suppose vecB is the query, which has smaller size
        if vecB.shape[0] > vecA.shape[0]:
            vecA, vecB = vecB, vecA
        
        vecB = vecB.cuda()
        
        result_tensors = []

        for batch in torch.split(vecA, 10000, dim=0):
            batch = batch.cuda()
            if self.distance_type == 'Tanimoto':
                results = -self._Tanimoto(batch, vecB)
            elif self.distance_type == 'Dice':
                results = -self._dice(batch, vecB)
            elif self.distance_type == 'Cosine':
                results = -self._cosine(batch, vecB)
            elif self.distance_type == 'Euclidean':
                results = self._euclidean(batch, vecB)
            elif self.distance_type == 'Sokal':
                results = -self._sokal(batch, vecB)
            elif self.distance_type == 'Russel':
                results = -self._russel(batch, vecB)
            elif self.distance_type == 'Kulczynski':
                results = -self._kulczynski(batch, vecB)
            elif self.distance_type == 'McConnaughey':
                results = -self._McConnaughey(batch, vecB)
            else:
                raise NotImplementedError
            
            result_tensors.append(results.cpu())
    
        return torch.cat(result_tensors, dim=0)
    
class Retriever:
    
    def __init__(self, 
                 embedding_type: str, 
                 distance_type: str, 
                 embedding_dir: str, 
                 smiles_lib_path: str, 
                 prunning: bool = False,
                 block_size: int = 1e6):
        self.embedding_type = embedding_type
        self.distance_type = distance_type
        self.embedding_dir = embedding_dir
        self.embedding = Embedding(embedding_type)
        self.distance = Distance(distance_type)
        if self.embedding_type != 'random':
            self.lib_files = [os.path.join(self.embedding_dir, self.embedding_type, x)
                                        for x in os.listdir(os.path.join(self.embedding_dir, self.embedding_type)) 
                                        if x.endswith('.pt')]
            self.lib_files = sorted(self.lib_files, key=lambda x: int(x.split('/')[-1].strip('.pt')))
            self.block_size = block_size
            # sanity check
            assert Embedding.load(self.lib_files[0]).shape[0] == self.block_size, "block size & lib_files not consistent!"
            
        self.smiles_lib = [str(x.strip().split()[0]) for x in open(smiles_lib_path, 'r').readlines()]
        self.size_lib = [int(x.strip().split()[1]) for x in open(smiles_lib_path, 'r').readlines()]
        
        self._retrieve = self._naive_retrieve if not prunning else self._prunning_retrieve
    
    
    def recieve_query(self, query_path):
        queries = [Chem.MolFromSmiles(x.strip()) for x in open(query_path, 'r').readlines()]
        print('{} queries loaded from {}'.format(len(queries), query_path))
        return queries
    
    def save_result(self, smiles_list_of_list, result_path):
        for i, smiles_list in enumerate(smiles_list_of_list):
            with open(os.path.join(result_path, f'{i}.txt'), 'w') as f:
                f.write('\n'.join(smiles_list))
        print('{} result saved to {}'.format(len(smiles_list_of_list), result_path))
        
    def _naive_retrieve(self, query: Chem.Mol, topk: int):
        query_embedding = self.embedding(query)
        retrieved = []
        
        for i, lib_file in enumerate(self.lib_files):
            # load embedding library
            embedding_block = Embedding.load(lib_file)
            # compute distance
            distance_block = self.distance.distance(query_embedding, embedding_block)
            # filter top_k molecules in this block
            id2distance = [(int(i*self.block_size+j), distance_block[j].item()) for j in range(len(distance_block))]
            topk_id2distance = heapq.nsmallest(topk, id2distance, key=lambda x: x[1])
            retrieved += topk_id2distance

        # get top_k molecules
        retrieved = heapq.nsmallest(topk, retrieved, key=lambda x: x[1])
        return [self.smiles_lib[i] for (i,j) in retrieved]

    def _prunning_retrieve(self, query: Chem.Mol, topk: int):
        
        query_size = query.GetNumAtoms()
        
        reversed_lib = self.size_lib[::-1]

        #######  reversed_lib: [query_size/1.5 ------- query_size*1.5] (upper_bound, lower_bound-1) #######
        
        upper_bound = max(0, bisect.bisect_left(reversed_lib, int(query_size / 1.5),))
        lower_bound = min(len(self.size_lib)-1, bisect.bisect_right(reversed_lib, int(query_size * 1.5)))

        lib_candidates = list(range(len(self.lib_files) - int(lower_bound // self.block_size) - 1, 
                                    len(self.lib_files) - int(upper_bound // self.block_size)))
        #print('search in', lib_candidates)
        
        query_embedding = self.embedding(query)
        retrieved = []
        
        for lib_candidate in lib_candidates:
            # load embedding library
            embedding_block = Embedding.load(self.lib_files[lib_candidate])
            # compute distance
            distance_block = self.distance.distance(query_embedding, embedding_block)
            # filter top_k molecules in this block
            id2distance = [(int(lib_candidate*self.block_size+j), distance_block[j].item()) for j in range(len(distance_block))]
            topk_id2distance = heapq.nsmallest(topk, id2distance, key=lambda x: x[1])
            retrieved += topk_id2distance

        # get top_k molecules
        retrieved = heapq.nsmallest(topk, retrieved, key=lambda x: x[1])
        return [self.smiles_lib[i] for (i,j) in retrieved[:topk]]
    
    def __call__(self, query: Union[Chem.Mol, List[Chem.Mol]], topk: int):
        
        if self.embedding_type == 'random':
            
            if isinstance(query, list):
                return [
                    [self.smiles_lib[random.randint(0, len(self.smiles_lib)-1)] for _ in range(topk)] for __ in range(len(query))
                ]
            elif isinstance(query, Chem.Mol):
                return [self.smiles_lib[random.randint(0, len(self.smiles_lib)-1)] for _ in range(topk)]
            else:
                raise NotImplementedError
            
        if isinstance(query, list):
            return [self._retrieve(q, topk) for q in query]
        elif isinstance(query, Chem.Mol):
            return self._retrieve(query, topk)
        else:
            raise NotImplementedError("query must be either a list of rdkit.Chem.Mol or rdkit.Chem.Mol")
    
if __name__ == '__main__':
    # test
    mol1 = [Chem.MolFromSmiles('O[C@H]1CN2CC[C@H](O)[C@@H]2[C@@H](O)[C@@H]1O'),Chem.MolFromSmiles('COc1ccc(CSC2=NCCN2)cc1Cl'), Chem.MolFromSmiles('CCc1ccc(C(F)F)c(C(N)=O)c1[N+](=O)[O-]')]  
    mol2 = Chem.MolFromSmiles('NCCCN(CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCCN(CCCN(CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN)CCCN(CCCN(CCCN)CCCN)CCCN(CCCN)CCCN')
    
    for embed_type in embeddingTypes:
        print("\n#### embedding:", embed_type)
        embedding = Embedding(embed_type)
        
        fp1_tensor = embedding(mol1)
        fp2_tensor = embedding(mol2)

        fps1 = [embeddingFunctions[embed_type](x) for x in mol1]
        fps2 = embeddingFunctions[embed_type](mol2)
        
        if embed_type == 'EStateFingerprint':
            continue
        
        for sim in similarityFunctions:
            
            gt = [similarityFunctions[sim](x, fps2) for x in fps1]
            
            predicted = Distance(sim).distance(fp1_tensor, fp2_tensor).squeeze()
            print(sim, predicted, gt)
        
        print(torch.norm(fp1_tensor - fp2_tensor, dim=1, p=2))
        print(Distance('Euclidean').distance(fp1_tensor, fp2_tensor))
        