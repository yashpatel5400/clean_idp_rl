import numpy as np
import torch
import os
import pickle

from conformer_rl import utils
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_chignolin import generate_chignolin
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit
from conformer_rl.agents import PPORecurrentExternalCurriculumAgent
from conformer_rl.utils import seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                     
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    utils.set_one_thread()

    # Create mol_configs for the curriculum
    chignolin_fasta = "YYDPETGTWY"
    curriculum_lens = [3, 5, 7, 10]

    mol_configs = []
    for curriculum_len in curriculum_lens:
        curriculum_fasta = chignolin_fasta[:curriculum_len]
        filename = f"{curriculum_fasta}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                mol_config = pickle.load(f)
        else:
            seed(curriculum_fasta)
            curriculum_mol = generate_chignolin(curriculum_fasta)
            mol_config = config_from_rdkit(curriculum_mol, num_conformers=1000, calc_normalizers=True) 
            mol_config.mol_name = curriculum_fasta
            with open(filename, "wb") as f:
                pickle.dump(mol_config, f)
        
        mol_configs.append(mol_config)
    
    chignolin_mol = generate_chignolin(chignolin_fasta)
    seed(chignolin_fasta)
    eval_mol_config = config_from_rdkit(chignolin_mol, num_conformers=1000, calc_normalizers=True)

    config = Config()
    config.tag = 'curriculum_chignolin'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Batch Hyperparameters
    config.max_steps = 200001

    # training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # Task Settings
    config.train_env = Task('GibbsScoreLogPruningCurriculumEnv-v0', concurrency=True, num_envs=10, seed=np.random.randint(0,1e5), mol_configs=mol_configs)
    config.eval_env = Task('GibbsScoreLogPruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=eval_mol_config)
    config.eval_interval = 20000
    config.eval_episodes = 2

    # curriculum Hyperparameters
    config.curriculum_agent_buffer_len = 20
    config.curriculum_agent_reward_thresh = 0.2
    config.curriculum_agent_success_rate = 0.7
    config.curriculum_agent_fail_rate = 0.2

    agent = PPORecurrentExternalCurriculumAgent(config)
    agent.run_steps()