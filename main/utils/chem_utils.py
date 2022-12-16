from deepchem.utils import conformers
import numpy as np
import bisect
import torch
import logging
import time
from tqdm import tqdm
import mdtraj as md

import openmm
import openmm.app as app
import openmm.unit as u

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Geometry import Point3D

from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import ForceField
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator

class MDSimulator:
    def __init__(self, rd_mol):
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(rd_mol)
        ofmol = Molecule.from_rdkit(rd_mol)
        ofmol.name = 'molecule'

        forcefield_kwargs = {
            'constraints' : app.HBonds, 
            'rigidWater' : True, 
            'removeCMMotion' : False, 
            'hydrogenMass' : 4 * u.amu 
        }
        
        off_top = ofmol.to_topology()
        omm_top = off_top.to_openmm()
        
        system_generator = SystemGenerator(small_molecule_forcefield='gaff-2.11', molecules=[ofmol], forcefield_kwargs=forcefield_kwargs, cache='db.json')
        system = system_generator.create_system(omm_top)
        
        integrator = openmm.VerletIntegrator(0.002 * u.picoseconds)
        platform = openmm.Platform.getPlatformByName("CUDA")
        # start at 1, since we assume 0 is being used by PyTorch, to avoid GPU memory issues
        assigned_gpu = np.random.randint(0, torch.cuda.device_count())
        prop = dict(CudaPrecision="mixed", DeviceIndex=f"{assigned_gpu}")
        self.simulator = app.Simulation(omm_top, system, integrator, platform, prop)
        
    def _np_to_mm(self, arr: np.ndarray, unit: openmm.unit=u.angstrom):
        wrapped_val = openmm.unit.quantity.Quantity(arr, unit)
        return wrapped_val

    def _init_simulator(self, mol, conf_id):
        conf = mol.GetConformer(conf_id)
        positions = self._np_to_mm(conf.GetPositions())
        self.simulator.context.setPositions(positions)
        self.simulator.context.setVelocitiesToTemperature(300 * u.kelvin)

    def optimize_conf(self, mol: Chem.Mol, conf_id: int = None):
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        self._init_simulator(mol, conf_id)
        
        # HACK: sometimes "openmm.OpenMMException: Particle coordinate is nan" so we skip those cases
        try:
            self.simulator.minimizeEnergy(maxIterations=500)
        except:
            return

        # OpenMM returns all of its positions in nm, so we have to convert back to Angstroms for RDKit
        optimized_positions_nm = self.simulator.context.getState(getPositions=True).getPositions()
        optimized_positions = optimized_positions_nm.in_units_of(u.angstrom) # match RDKit/MMFF convention

        conf = mol.GetConformer(conf_id)
        for i, pos in enumerate(optimized_positions):
            conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))

    def optimize_confs(self, mol: Chem.Mol):
        for conf_id in range(mol.GetNumConformers()):
            self.optimize_conf(mol, conf_id)

    def get_conformer_energy(self, mol: Chem.Mol, conf_id: int = None):
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        self._init_simulator(mol, conf_id)
        energy_kj = self.simulator.context.getState(getEnergy=True).getPotentialEnergy()
        energy_kcal = energy_kj.in_units_of(u.kilocalories_per_mole) # match RDKit/MMFF convention
        return energy_kcal._value

    def get_conformer_energies(self, mol: Chem.Mol) -> List[float]:
        """Returns a list of energies for each conformer in `mol`.
        """
        energies = []
        for conf in mol.GetConformers():
            energy = self.get_conformer_energy(mol, conf.GetId())
            energies.append(energy)
        
        return np.asarray(energies, dtype=float)

    def prune_conformers(self, mol: Chem.Mol, tfd_thresh: float, invalid_thresh = None) -> Chem.Mol:
        """Prunes all the conformers in the molecule.

        Removes conformers that have a TFD (torsional fingerprint deviation) lower than
        `tfd_thresh` with other conformers. Lowest energy conformers are kept.

        Parameters
        ----------
        mol : RDKit Mol
            The molecule to be pruned.
        tfd_thresh : float
            The minimum threshold for TFD between conformers.
        invalid_thresh : float
            The threshold for the energy to deem a conformer as being "invalid." (Note: this is a hack to
            get around the fact FFs return extremely low energies for some invalid states)

        Returns
        -------
        mol : RDKit Mol
            The updated molecule after pruning.
        """
        if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
            return mol

        energies = self.get_conformer_energies(mol)
        tfd = tfd_matrix(mol)
        sort = np.argsort(energies)  # sort by increasing energy
        keep = []  # always keep lowest-energy conformer
        discard = []

        for i in sort:
            this_tfd = tfd[i][np.asarray(keep, dtype=int)]        
            # discard conformers within the tfd threshold
            if np.all(this_tfd >= tfd_thresh) and (invalid_thresh is None or energies[i] > invalid_thresh):
                keep.append(i)
            else:
                discard.append(i)

        # create a new molecule to hold the chosen conformers
        # this ensures proper conformer IDs and energy-based ordering
        new = Chem.Mol(mol)
        new.RemoveAllConformers()

        for i in keep:
            conf = mol.GetConformer(int(i))
            new.AddConformer(conf, assignId=True)

        return new

class MDSimulatorPDB(MDSimulator):
    def __init__(self, rd_pdb):
        pdb = app.pdbfile.PDBFile(rd_pdb)
        forcefield = app.forcefield.ForceField("amber14/protein.ff14SBonlysc.xml", "implicit/gbn2.xml")
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod = app.forcefield.NoCutoff, constraints = app.forcefield.HBonds)
        integrator = openmm.VerletIntegrator(0.002 * u.picoseconds)
        platform = openmm.Platform.getPlatformByName("CUDA")
        # start at 1, since we assume 0 is being used by PyTorch, to avoid GPU memory issues
        assigned_gpu = np.random.randint(1, torch.cuda.device_count())
        prop = dict(CudaPrecision="mixed", DeviceIndex=f"{assigned_gpu}")
        self.simulator = app.Simulation(pdb.topology, system, integrator, platform, prop)

# class ConformerGeneratorCustom(conformers.ConformerGenerator):
#     # pruneRmsThresh=-1 means no pruning done here
#     # I don't use embed_molecule() because it does AddHs() & EmbedMultipleConfs()
#     def __init__(self, *args, **kwargs):
#         super(ConformerGeneratorCustom, self).__init__(*args, **kwargs)


#     # add progress bar
#     def minimize_conformers(self, mol):
#         """
#         Minimize molecule conformers.

#         Parameters
#         ----------
#         mol : RDKit Mol
#                 Molecule.
#         """
#         pbar = tqdm(total=mol.GetNumConformers())
#         for conf in mol.GetConformers():
#             ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
#             ff.Minimize()
#             pbar.update(1)
#         pbar.close()

#     def prune_conformers(self, mol, rmsd, heavy_atoms_only=True):
#         """
#         Prune conformers from a molecule using an RMSD threshold, starting
#         with the lowest energy conformer.

#         Parameters
#         ----------
#         mol : RDKit Mol
#                 Molecule.

#         Returns
#         -------
#         new: A new RDKit Mol containing the chosen conformers, sorted by
#                  increasing energy.
#         new_rmsd: matrix of conformer-conformer RMSD
#         """
#         if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
#             return mol
#         energies = self.get_conformer_energies(mol)
#     #     rmsd = get_conformer_rmsd_fast(mol)

#         sort = np.argsort(energies)  # sort by increasing energy
#         keep = []  # always keep lowest-energy conformer
#         discard = []

#         for i in sort:
#             # always keep lowest-energy conformer
#             if len(keep) == 0:
#                 keep.append(i)
#                 continue

#             # discard conformers after max_conformers is reached
#             if len(keep) >= self.max_conformers:
#                 discard.append(i)
#                 continue

#             # get RMSD to selected conformers
#             this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

#             # discard conformers within the RMSD threshold
#             if np.all(this_rmsd >= self.rmsd_threshold):
#                 keep.append(i)
#             else:
#                 discard.append(i)

#         # create a new molecule to hold the chosen conformers
#         # this ensures proper conformer IDs and energy-based ordering
#         new = Chem.Mol(mol)
#         new.RemoveAllConformers()
#         conf_ids = [conf.GetId() for conf in mol.GetConformers()]
#         for i in keep:
#             conf = mol.GetConformer(conf_ids[i])
#             new.AddConformer(conf, assignId=True)

#         new_rmsd = get_conformer_rmsd_fast(new, heavy_atoms_only=heavy_atoms_only)
#         return new, new_rmsd

def prune_last_conformer(mol, tfd_thresh, energies=None, quick=False):
    """
    Checks that most recently added conformer meats TFD threshold.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    tfd_thresh : TFD threshold
    energies: energies of all conformers minus the last one
    Returns
    -------
    new: A new RDKit Mol containing the chosen conformers, sorted by
             increasing energy.
    """

    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    idx = bisect.bisect(energies[:-1], energies[-1])

    tfd = Chem.TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        new_energys = list(range(0, mol.GetNumConformers() - 1))
        mol.RemoveConformer(mol.GetNumConformers() - 1)

        logging.debug('tossing conformer')

        return mol, new_energys


    else:
        logging.debug('keeping conformer', idx)
        keep = list(range(0,idx))
        # print('keep 1', keep)
        keep += [mol.GetNumConformers() - 1]
        # print('keep 2', keep)

        l = np.array(range(idx, len(tfd)))
        # print('L 1', l)
        # print('tfd', tfd)
        l = l[tfd[idx:] >= tfd_thresh]
        # print('L 2', l)

        keep += list(l)
        # print('keep 3', keep)

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]

        for i in keep:
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)

        return new, keep

def tfd_matrix(mol: Chem.Mol) -> np.array:
    """Calculates the TFD matrix for all conformers in a molecule.
    """
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix

def prune_last_conformer_quick(mol, tfd_thresh, energies=None):
    """
    Checks that most recently added conformer meats TFD threshold.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    tfd_thresh : TFD threshold
    energies: energies of all conformers minus the last one
    Returns
    -------
    new: A new RDKit Mol containing the chosen conformers, sorted by
             increasing energy.
    """

    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    tfd = Chem.TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    if not np.all(tfd >= tfd_thresh):
        logging.debug('tossing conformer')
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, 0.0
    else:
        logging.debug('keeping conformer')
        return mol, 1.0



# def prune_conformers(mol, tfd_thresh, rmsd=False):
#     """
#     Prune conformers from a molecule using an TFD/RMSD threshold, starting
#     with the lowest energy conformer.

#     Parameters
#     ----------
#     mol : RDKit Mol
#             Molecule.
#     tfd_thresh : TFD threshold
#     Returns
#     -------
#     new: A new RDKit Mol containing the chosen conformers, sorted by
#              increasing energy.
#     """

#     confgen = ConformerGeneratorCustom()

#     if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
#         return mol

#     energies = confgen.get_conformer_energies(mol)

#     if not rmsd:
#         tfd = array_to_lower_triangle(Chem.TorsionFingerprints.GetTFDMatrix(mol, useWeights=False), True)
#     else:
#         tfd = get_conformer_rmsd_fast(mol)
#     sort = np.argsort(energies)  # sort by increasing energy
#     keep = []  # always keep lowest-energy conformer
#     discard = []

#     for i in sort:
#         # always keep lowest-energy conformer
#         if len(keep) == 0:
#             keep.append(i)
#             continue

#         # get RMSD to selected conformers
#         this_tfd = tfd[i][np.asarray(keep, dtype=int)]
#         # discard conformers within the RMSD threshold
#         if np.all(this_tfd >= tfd_thresh):
#             keep.append(i)
#         else:
#             discard.append(i)

#     # create a new molecule to hold the chosen conformers
#     # this ensures proper conformer IDs and energy-based ordering
#     new = Chem.Mol(mol)
#     new.RemoveAllConformers()
#     conf_ids = [conf.GetId() for conf in mol.GetConformers()]
#     for i in keep:
#         conf = mol.GetConformer(conf_ids[i])
#         new.AddConformer(conf, assignId=True)

#     return new

def print_torsions(mol):
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    conf = mol.GetConformer(id=0)
    tups = [atoms[0] for atoms, ang in nonring]
    degs = [Chem.rdMolTransforms.GetDihedralDeg(conf, *tup) for tup in tups]
    print(degs)

# def print_energy(mol):
#     confgen = ConformerGeneratorCustom(max_conformers=1,
#                                  rmsd_threshold=None,
#                                  force_field='mmff',
#                                  pool_multiplier=1)
#     print(confgen.get_conformer_energies(mol))

def array_to_lower_triangle(arr, get_symm=False):
    # convert list to lower triangle mat
    n = int(np.sqrt(len(arr)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    lt_mat = np.zeros((n,n))
    lt_mat[idx] = arr
    if get_symm == True:
        return lt_mat + np.transpose(lt_mat) # symmetric matrix
    return lt_mat
