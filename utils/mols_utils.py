from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
import logging

def generate_conformer(mol):
    ps = AllChem.ETKDGv3()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        logging.warning('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        logging.warning("use ETDKGv3 to generate conformations")
    
    return mol