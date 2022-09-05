import os
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import pickle
import time
import subprocess
import gpu_check

# Create config object
base_path = "/home/yppatel/"
drugs_file = os.path.join(base_path, "rdkit_folder/summary_drugs.json")

with open(drugs_file, "r") as f:
    drugs_summ = json.load(f)

results_fns = os.listdir(os.path.join(base_path, "conformer-rl/examples/data/geom_results"))
completed_mols = set([int(results_fn.split("_")[1]) for results_fn in results_fns if "mol" in results_fn])

num_test_drugs = 1000
total_num_drugs = (len(list(drugs_summ.keys())))

mol_ids_pkl_fn = os.path.join(base_path, "conformer-rl/examples/mol_ids.pickle")

if not os.path.exists(mol_ids_pkl_fn):
    mol_ids = np.random.randint(0, total_num_drugs, size=num_test_drugs)
    mol_ids = [mol_id for mol_id in mol_ids if mol_id not in completed_mols]
    with open(mol_ids_pkl_fn, "wb") as f:
        pickle.dump(mol_ids, f)

with open(mol_ids_pkl_fn, "rb") as f:
    mol_ids = pickle.load(f)
 
scheduling_thresh = 3000

for mol_id in mol_ids:
    while True:
        gpus = gpu_check.getGPUs()
        found = False
        for gpu_idx, gpu in enumerate(gpus):
            if gpu.memoryFree > scheduling_thresh:
                bashCommand = f"CUDA_VISIBLE_DEVICES={gpu_idx} PYTHONPATH=/home/yppatel/conformer-rl/src/ python geom_run.py --mol_id {mol_id}"
                print(bashCommand)
                
                subprocess.Popen([bashCommand], shell=True, stdin=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(20) # wait for GPU scheduling to occur (compute normalizers)
                found = True
                break
        
        if not found:
            time.sleep(10)
        else:
            break