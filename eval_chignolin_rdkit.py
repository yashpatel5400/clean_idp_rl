import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import random
import time
import os
import json
from tempfile import TemporaryDirectory
import subprocess
from concurrent.futures import ProcessPoolExecutor

from main.utils import *

def run_chignolin_rdkit(tup):
    smiles, energy_norm, gibbs_norm = tup

    # mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    mol = Chem.rdmolfiles.MolFromPDBFile("/home/yppatel/misc/clean_idp_rl/chignolin/YYDPETGTWY.pdb", removeHs=False)

    start = time.time()

    res = AllChem.EmbedMultipleConfs(mol, numConfs=200, numThreads=-1)
    md_sim = MDSimulatorPDB("/home/yppatel/misc/clean_idp_rl/chignolin/YYDPETGTWY.pdb")
    res = md_sim.optimize_confs(mol)
    mol = md_sim.prune_conformers(mol, 0.05)

    energys = md_sim.get_conformer_energies(mol)
    total = np.sum(np.exp(-(energys-energy_norm)))
    total /= gibbs_norm
    end = time.time()
    return total, end-start


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    outputs = []
    times = []

    chignolin_args = ( "[H]OC1=C([H])C([H])=C(C([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N2C([H])([H])C([H])([H])C([H])([H])C2([H])C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])([H])C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(O)O)C([H])([H])C2=C([H])C([H])=C(O[H])C([H])=C2[H])C([H])([H])C2=C([H])N([H])C3=C2C([H])=C([H])C([H])=C3[H])C([H])(O[H])C([H])([H])[H])C([H])(O[H])C([H])([H])[H])C([H])([H])C([H])([H])C(=O)O)C([H])([H])C(=O)O)N([H])C(=O)C([H])(C([H])([H])C2=C([H])C([H])=C(O[H])C([H])=C2[H])[N+]([H])([H])[H])C([H])=C1[H]", -153.87714114204206, 1.1194140820851908)
    
    args_list = [chignolin_args] * 10
    with ProcessPoolExecutor() as executor:
        out = executor.map(run_chignolin_rdkit, args_list)

    for a, b in out:
        outputs.append(a)
        times.append(b)

    print('outputs', outputs)
    print('mean', np.array(outputs).mean())
    print('std', np.array(outputs).std())
    print('times', times)
