"""
Chignolin Generator
=================
"""
from rdkit import Chem

def generate_chignolin(chunk) -> Chem.Mol:
    """Generates chignolin molecule.
    """

    chignolin_pdb_fn = f"src/conformer_rl/molecule_generation/chignolin/{chunk}.pdb"
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    Chem.SanitizeMol(chignolin)
    return chignolin