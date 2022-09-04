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

def calc_performance_stats(true_mol, model_mol, thresholds):
    num_true_confs = true_mol.GetNumConformers()
    num_model_confs = model_mol.GetNumConformers()

    rmsd_list = []
    for true_conf_id in range(num_true_confs):
        for model_conf_id in range(num_model_confs):
            try:
                rmsd_val = AllChem.GetBestRMS(true_mol, model_mol, prbId=true_conf_id, refId=model_conf_id)
            except RuntimeError:
                return None
            rmsd_list.append(rmsd_val)

    rmsd_array = np.array(rmsd_list).reshape(num_true_confs, num_model_confs)

    coverage_recall = np.sum(rmsd_array.min(axis=1, keepdims=True) < thresholds, axis=0) / num_true_confs
    amr_recall = rmsd_array.min(axis=1).mean()

    coverage_precision = np.sum(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(thresholds, 1), axis=1) / num_model_confs
    amr_precision = rmsd_array.min(axis=0).mean()
    
    return coverage_recall, amr_recall, coverage_precision, amr_precision


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        for results_fn in iter(self.queue.get, None):
            mol_id = int(os.path.basename(results_fn).split("_")[1].replace(".pdb",""))
            print(f"Starting mol {mol_id}...")

            metrics_dir = os.path.join(base_path, "conformer-rl/examples/data/geom_metrics/diff/")
            metrics_fn = os.path.join(metrics_dir, f"{mol_id}.pickle")
            
            example_smiles = list(drugs_summ.keys())[mol_id]
            mol_path = drugs_summ[example_smiles]["pickle_path"]
            with open(os.path.join(base_path, "rdkit_folder", mol_path), "rb") as f:
                mol = pickle.load(f)

            base_mol = mol["conformers"][0]["rd_mol"]
            ref_mol = copy.deepcopy(base_mol)
            [ref_mol.AddConformer(mol["rd_mol"].GetConformer(0), assignId=True) for mol in mol["conformers"][1:]]

            print(f"Computing metrics for mol {mol_id}...")
            pred_mol = Chem.rdmolfiles.MolFromPDBFile(results_fn, removeHs=False)
            thresholds = np.arange(0, 2.5, .125)
            metrics = calc_performance_stats(ref_mol, pred_mol, thresholds)

            print(f"Completed mol {mol_id}!")
            with open(metrics_fn, "wb") as f:
                pickle.dump(metrics, f)

request_queue = Queue()
num_workers = 32
for i in range(num_workers):
    Worker(request_queue).start()

results_dir = os.path.join(base_path, "conformer-rl/examples/data/geom_diff")
results_fns = os.listdir(results_dir)
for results_fn in results_fns:
    request_queue.put(os.path.join(results_dir, results_fn))

# Sentinel objects to allow clean shutdown: 1 per worker.
for i in range(num_workers):
    request_queue.put(None) 