import copy
import os
import json
import random
import rdkit
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from multiprocessing import Queue, Process

methods = ["rl", "diff", "etkdg"]

def calc_performance_stats(true_mol, model_mol, thresholds):
    num_true_confs = true_mol.GetNumConformers()
    num_model_confs = model_mol.GetNumConformers()

    print(f"true_confs={num_true_confs} model_confs={num_model_confs}")

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
        for common_mol_fn in iter(self.queue.get, None):
            for method in methods:
                mol_id = int(common_mol_fn.split("_")[1].replace(".pdb",""))
                print(f"Computing metrics for mol {mol_id}...")
                
                metrics_dir = os.path.join(base_path, f"conformer-rl/examples/data/geom_metrics/{method}")
                metrics_fn = os.path.join(metrics_dir, f"{mol_id}.pickle")

                if os.path.exists(metrics_fn):
                    continue
                
                true_mol_fn = os.path.join(base_path, "conformer-rl/examples/data/", f"geom_truth_pruned", common_mol_fn)
                method_mol_fn = os.path.join(base_path, "conformer-rl/examples/data/", f"geom_{method}_pruned", common_mol_fn)

                ref_mol = Chem.rdmolfiles.MolFromPDBFile(true_mol_fn, removeHs=False)
                pred_mol = Chem.rdmolfiles.MolFromPDBFile(method_mol_fn, removeHs=False)

                thresholds = np.arange(0, 2.5, .125)
                metrics = calc_performance_stats(ref_mol, pred_mol, thresholds)

                print(f"Completed mol {mol_id}!")
                with open(metrics_fn, "wb") as f:
                    pickle.dump(metrics, f)

base_path = "/home/yppatel/"

method_mol_fns = []
for method in methods:
    results_folder = os.path.join(base_path, "conformer-rl/examples/data/", f"geom_{method}_pruned")
    method_mol_fns.append(set(os.listdir(results_folder)))
common_mol_fns = set.intersection(*method_mol_fns)

request_queue = Queue()
num_workers = 16
for i in range(num_workers):
    Worker(request_queue).start()

for common_mol_fn in common_mol_fns:
    request_queue.put(common_mol_fn)

# Sentinel objects to allow clean shutdown: 1 per worker.
for i in range(num_workers):
    request_queue.put(None) 