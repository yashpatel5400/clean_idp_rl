import os
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_alkanes import generate_branched_alkane
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

import logging
logging.basicConfig(level=logging.DEBUG)

import json
import pickle
import copy

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_id", type=int)
    args = parser.parse_args()

    train_seed = 30000
    eval_seed  = 40000

    utils.set_one_thread()

    # Create config object
    base_path = "/home/yppatel/"
    drugs_file = os.path.join(base_path, "rdkit_folder/summary_drugs.json")

    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    example_smiles = list(drugs_summ.keys())[args.mol_id]
    mol_path = drugs_summ[example_smiles]["pickle_path"]
    with open(os.path.join(base_path, "rdkit_folder", mol_path), "rb") as f:
        mol_props = pickle.load(f)

    base_mol = mol_props["conformers"][0]["rd_mol"]
    mol = copy.deepcopy(base_mol)
    mol.RemoveAllConformers()
    
    mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file=f'mol_{args.mol_id}')

    # Create agent training config object
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Config()
    config.device = device

    config.tag = f'mol_{args.mol_id}'
    config.max_steps = 40001

    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Logging Parameters
    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

    # Configure Environment
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=5, seed=train_seed, mol_config=mol_config)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=eval_seed, mol_config=mol_config)
    config.eval_interval = 20000
    config.eval_episodes = 2

    agent = PPORecurrentAgent(config)
    
    agent.run_steps()