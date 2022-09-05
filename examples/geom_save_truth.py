import copy
import os
import json
import random
import rdkit
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Queue, Process

import sys
sys.path.append("../src/")
from conformer_rl import analysis
from conformer_rl import utils

sns.set_style("dark")
plt.rcParams['text.usetex'] = True
plt.rcParams["axes.grid"] = False

base_path = "/home/yppatel/"
drugs_file = os.path.join(base_path, "rdkit_folder/summary_drugs.json")

with open(drugs_file, "r") as f:
    drugs_summ = json.load(f)

results_fns = os.listdir(os.path.join(base_path, "conformer-rl/examples/data/geom_results"))

for results_fn in results_fns:
    mol_id = int(results_fn.split("_")[1])
    print(f"Computing metrics for mol {mol_id}...")

    metrics_dir = os.path.join(base_path, "conformer-rl/examples/data/geom_truth/")
    pdb_fn = os.path.join(metrics_dir, f"{mol_id}.pdb")
    # if os.path.exists(pdb_fn):
    #     continue
            
    mol_smile = list(drugs_summ.keys())[mol_id]
    mol_path = drugs_summ[mol_smile]["pickle_path"]
    with open(os.path.join(base_path, "rdkit_folder", mol_path), "rb") as f:
        mol_props = pickle.load(f)

    base_mol = mol_props["conformers"][0]["rd_mol"]
    ref_mol = copy.deepcopy(base_mol)
    [ref_mol.AddConformer(mol["rd_mol"].GetConformer(0), assignId=True) for mol in mol_props["conformers"][1:25]]

    Chem.rdmolfiles.MolToPDBFile(ref_mol, pdb_fn)