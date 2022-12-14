import time
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import json
import tqdm
import numpy as np
from math import comb, ceil
import deepchem

import multiprocessing
from multiprocessing.pool import Pool

import random
from concurrent.futures import ProcessPoolExecutor

from main.utils import *

def create_chignolin(mol_fn, out_dir):
    result_fn = os.path.join(out_dir, f'{os.path.basename(mol_fn).split(".")[0]}.json')
    if os.path.exists(result_fn):
        print(f"Already have: {result_fn}! Skipping...")
        return
    print(f"Creating: {result_fn}...")

    m = Chem.rdmolfiles.MolFromPDBFile(mol_fn, removeHs=False)
    md_sim = MDSimulatorPDB(mol_fn)
    AllChem.EmbedMultipleConfs(m, numConfs=1000, numThreads=-1)
    md_sim.optimize_confs(m)

    m = md_sim.prune_conformers(m, 0.05)

    energys = md_sim.get_conformer_energies(m)
    print(len(TorsionFingerprints.CalculateTorsionLists(m)[0]))
    standard = energys.min()
    total = np.sum(np.exp(-(energys-standard)))

    nonring, ring = Chem.TorsionFingerprints.CalculateTorsionLists(m)
    rbn = len(nonring)
    out = {
        'molfile': os.path.basename(mol_fn),
        'standard': standard,
        'total': total
    }

    with open(result_fn, 'w') as fp:
        json.dump(out, fp)

def create_chignolin_wrapper(args):
    return create_chignolin(*args)

def generate_pdb_from_fasta(in_dir, full_fasta, curriculum_stages = None):
    if curriculum_stages is None:
        curriculum_stages = []
        for i in range(2, len(full_fasta) + 1):
            for starting_point in range(len(full_fasta) - i):
                curriculum_stages.append((starting_point, starting_point+i))
    
    for curriculum_stage in curriculum_stages:
        starting_point, ending_point = curriculum_stage
        sub_fasta = full_fasta[starting_point:ending_point]
        dest_fn = os.path.join(in_dir, f"{sub_fasta}.pdb")
        if os.path.exists(dest_fn):
            print(f"Already have: {sub_fasta}! Skipping...")
            continue

        mol = Chem.rdmolfiles.MolFromFASTA(sub_fasta)
        AllChem.EmbedMolecule(mol)
        hydrogenated_mol = deepchem.utils.rdkit_utils.add_hydrogens_to_mol(mol, is_protein=True)
        Chem.rdmolfiles.MolToPDBFile(hydrogenated_mol, dest_fn)
        print(f"Completed: {sub_fasta}")

if __name__ == "__main__":
    in_dir = "/home/yppatel/misc/clean_idp_rl/disordered_chignolin"
    out_dir = "/home/yppatel/misc/clean_idp_rl/disordered_chignolin_out"

    full_fasta = "GYDPETGTWG"
    generate_from_fastas = True
    if generate_from_fastas:
        curriculum_stages = [(None, 3), (None, 5), (None, 7), (None, 10)]
        generate_pdb_from_fasta(in_dir, full_fasta, curriculum_stages = curriculum_stages)

    fns = os.listdir(in_dir)
    full_fns = [(os.path.join(in_dir, fn), out_dir,) for fn in fns]

    multiprocessing.set_start_method('spawn')
    p = Pool(multiprocessing.cpu_count())
    p.map(create_chignolin_wrapper, full_fns)
    p.close()
    p.join()