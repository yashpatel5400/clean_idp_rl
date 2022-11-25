import time
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import json
import tqdm
import numpy as np
from math import comb, ceil
import deepchem

import random
from concurrent.futures import ProcessPoolExecutor

from main.utils import *

def create_chignolin(mol_fn, out_dir):
    m = Chem.rdmolfiles.MolFromPDBFile(mol_fn, removeHs=False)
    md_sim = MDSimulatorPDB(mol_fn)
    AllChem.EmbedMultipleConfs(m, numConfs=200, numThreads=-1)
    md_sim.optimize_confs(m)

    m = md_sim.prune_conformers(m, 0.05)

    energys = md_sim.get_conformer_energies(m)
    print(len(TorsionFingerprints.CalculateTorsionLists(m)[0]))
    standard = energys.min()
    total = np.sum(np.exp(-(energys-standard)))

    nonring, ring = Chem.TorsionFingerprints.CalculateTorsionLists(m)
    rbn = len(nonring)
    out = {
        'mol': Chem.MolToSmiles(m, isomericSmiles=False),
        'standard': standard,
        'total': total
    }

    with open(os.path.join(out_dir, f'{os.path.basename(mol_fn).split(".")[0]}.json'), 'w') as fp:
        json.dump(out, fp)

in_dir = "/home/yppatel/misc/clean_idp_rl/chignolin_granular"
out_dir = "/home/yppatel/misc/clean_idp_rl/chignolin_granular_out"

generate_from_fastas = True
if generate_from_fastas:
    full_fasta = "YYDPETGTWY"
    for i in range(2, len(full_fasta) + 1):
        for starting_point in range(len(full_fasta) - i):
            sub_fasta = full_fasta[starting_point:starting_point+i]
            dest_fn = os.path.join(in_dir, f"{sub_fasta}.pdb")
            if os.path.exists(dest_fn):
                print(f"Already have: {sub_fasta}! Skipping...")
                continue

            mol = Chem.rdmolfiles.MolFromFASTA(sub_fasta)
            AllChem.EmbedMolecule(mol)
            hydrogenated_mol = deepchem.utils.rdkit_utils.add_hydrogens_to_mol(mol, is_protein=True)
            Chem.rdmolfiles.MolToPDBFile(hydrogenated_mol, dest_fn)
            print(f"Completed: {sub_fasta}")

fns = os.listdir(in_dir)
full_fns = [os.path.join(in_dir, fn) for fn in fns]

for full_fn in full_fns:
    create_chignolin(full_fn, out_dir)