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

def run_lignins_rdkit(tup):
    smiles, energy_norm, gibbs_norm = tup

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    start = time.time()

    res = AllChem.EmbedMultipleConfs(mol, numConfs=200, numThreads=-1)
    md_sim = MDSimulator(mol)
    res = md_sim.optimize_confs(mol)
    mol = md_sim.prune_conformers(mol, 0.05)

    energys = md_sim.get_conformer_energies(mol)
    total = np.sum(np.exp(-(energys-energy_norm)))
    total /= gibbs_norm
    end = time.time()
    return total, end-start


if __name__ == '__main__':
    outputs = []
    times = []

    eleven_alkane_args = ( "[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]", 7.840935037731404,  13.066560104213275)
    trihexyl_args = ('[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[C@@]([H])(C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]', 14.88278294332602, 1.2363186365185044)

    args_list = [trihexyl_args] * 10
    with ProcessPoolExecutor() as executor:
        out = executor.map(run_lignins_rdkit, args_list)

    for a, b in out:
        outputs.append(a)
        times.append(b)

    print('outputs', outputs)
    print('mean', np.array(outputs).mean())
    print('std', np.array(outputs).std())
    print('times', times)
