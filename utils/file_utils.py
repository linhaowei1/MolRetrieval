import os
from rdkit import Chem

def tsv_reader(fn):
    '''
        read a tsv file. return a list of lists
    '''
    data = []
    with open(fn, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            data.append(fields)
    # remove header
    return data[1:]

def sdf_reader(fn):
    '''
        read a sdf file. return a rdmol object
    '''
    ligand_rdmol = next(iter(Chem.SDMolSupplier(fn)))
    #ligand_rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(ligand_rdmol))
    return ligand_rdmol

if __name__ == '__main__':
    mol = sdf_reader('/home/haowei/Desktop/repos/DSR/benchmarks/crossdocked/data/0/ligand.sdf')
    import pdb; pdb.set_trace()