"""
Curriculum Conformer_env
========================
"""

import logging
from typing import List
import copy

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import TorsionFingerprints
import gym

from conformer_rl.config import MolConfig
from conformer_rl.environments.conformer_env import ConformerEnv
from conformer_rl.utils import seed

class CurriculumConformerEnv(ConformerEnv):
    """Base interface for building conformer generation environments with support for curriculum learning.
    
    Parameters
    ----------
    mol_configs : list of :class:`~conformer_rl.config.mol_config.MolConfig`
        List of configuration object specifying the molecules and their corresponding parameters to be trained on
        as part of the curriculum. The list should be sorted in order of increasing task difficulty.

    Attributes
    ----------
    configs : list of :class:`~conformer_rl.config.mol_config.MolConfig`
        Configuration objects specifying molecules and corresponding parameters to be used in the environment,
        in the order of the designated curriculum (ordered from least to most difficult).
    total_reward : float
        Keeps track of the total reward for the current episode.
    current_step : int
        Keeps track of the number of elapsed steps in the current episode.
    step_info : dict from str to list
        Used for keeping track of data obtained at each step of an episode for logging.
    episode_info : dict from str to Any
        Used for keeping track of data useful at the end of an episode, such as total_reward, for logging.
    curriculum_max_index : int
        One plus the maximum index in which a molecule/task from the input list of ``mol_configs`` can be selected to be trained on.
        This attribute will be increased as the agent gets better at the current tasks in the curriculum and is ready to move on to
        more difficult tasks.
    """

    def __init__(self, mol_configs: List[MolConfig], tag = None):
        gym.Env.__init__(self)
        logging.debug('initializing curriculum conformer environment')
        self.configs = copy.deepcopy(mol_configs)
        self.curriculum_max_index = 1

        self.config = self.configs[0]
        self.tag = tag

        self.mol = self.config.mol
        self.mol.RemoveAllConformers()
        if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
            raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()
        self.setup_torsion_angles()

        self.reset()

    def reset(self) -> object:
        """Resets the environment and returns the observation of the environment.
        """
        self.total_reward = 0
        self.current_step = 0
        self.step_info = {}
        self.episode_info = {}

        # set index for the next molecule based on curriculum
        if self.curriculum_max_index == 1:
            index = 0
        else:
            p = 0.5 * np.ones(self.curriculum_max_index) / (self.curriculum_max_index - 1)
            p[-1] = 0.5
            index = np.random.choice(self.curriculum_max_index, p=p)

        with open(f"steps_{self.tag}.txt", "a") as f:
            f.writelines([f'Current Curriculum Molecule Index: {index}\n'])

        # set up current molecule
        mol_config = self.configs[index]
        seed(mol_config.mol_name)
        self.config = mol_config
        self.max_steps = mol_config.num_conformers
        self.mol = mol_config.mol
        self.mol.RemoveAllConformers()
        if Chem.EmbedMolecule(self.mol, randomSeed=self.config.seed, useRandomCoords=True) == -1:
            raise Exception('Unable to embed molecule with conformer using rdkit')
        self.conf = self.mol.GetConformer()
        self.setup_torsion_angles()

        self.episode_info['mol'] = Chem.Mol(self.mol)
        self.episode_info['mol'].RemoveAllConformers()
        self.episode_info['curriculum_level'] = self.curriculum_max_index
        
        obs = self._obs()
        return obs

    def setup_torsion_angles(self):
        [self.mol.GetAtomWithIdx(i).SetProp("original_index", str(i)) for i in range(self.mol.GetNumAtoms())]
        stripped_mol = Chem.rdmolops.RemoveHs(self.mol)

        nonring, _ = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
            
        original_to_stripped = {
            int(stripped_mol.GetAtomWithIdx(reindex).GetProp("original_index")) : reindex 
            for reindex in range(stripped_mol.GetNumAtoms())
        }
        self.nonring_reindexed = [
            [original_to_stripped[original] for original in atom_group] 
            for atom_group in self.nonring
        ]

    def increase_level(self):
        """Updates the ``curriculum_max_index`` attribute after obtaining signal from the agent that a favorable
        reward threshold has been achieved.
        """
        self.curriculum_max_index = min(self.curriculum_max_index * 2, len(self.configs))

    def decrease_level(self):
        """Updates the ``curriculum_max_index`` attribute after obtaining signal that the agent is performing
        poorly on the current curriclum range.
        """
        if self.curriculum_max_index > 1:
            self.curriculum_max_index = self.curriculum_max_index // 2