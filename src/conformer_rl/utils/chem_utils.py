"""
Chemistry Utilities
===================

Chemistry and molecule utility functions.
"""
import torch
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u
import bisect
from rdkit.Chem import TorsionFingerprints
import rdkit.Chem.AllChem as Chem
from rdkit.Geometry import Point3D

from typing import Tuple, List
import logging

simulator = None

def seed(mol_name):
    # toppar = [
    #     "src/conformer_rl/environments/environment_components/toppar/par_all36_prot.prm", 
    #     "src/conformer_rl/environments/environment_components/toppar/top_all36_prot.rtf",
    #     "src/conformer_rl/environments/environment_components/toppar/toppar_water_ions.str",
    # ]
    # psf = f"src/conformer_rl/molecule_generation/chignolin/{mol_name}.psf" # TODO: expand to non-chignolin

    # openmm_toppar = app.CharmmParameterSet(*toppar)
    # openmm_psf = app.CharmmPsfFile(psf)
    # openmm_system = openmm_psf.createSystem(openmm_toppar)

    # integrator = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.004 * u.picoseconds)

    pdb = app.pdbfile.PDBFile(f"src/conformer_rl/molecule_generation/chignolin/{mol_name}.pdb")
    forcefield = app.forcefield.ForceField("amber99sbildn.xml", "tip3p.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod = app.forcefield.NoCutoff, constraints = app.forcefield.HBonds)
    integrator = openmm.VerletIntegrator(0.002 * u.picoseconds)

    global simulator
    if torch.cuda.is_available():
        platform = openmm.Platform.getPlatformByName("CUDA")
        prop = dict(CudaPrecision="mixed", DeviceIndex="0")#, DisablePmeStream="true")
        simulator = app.Simulation(pdb.topology, system, integrator, platform, prop)
        # simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform, prop)
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        simulator = app.Simulation(pdb.topology, system, integrator, platform, prop)
        # simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform)

def np_to_mm(arr: np.ndarray, unit: openmm.unit=u.angstrom):
    wrapped_val = openmm.unit.quantity.Quantity(arr, unit)
    return wrapped_val

def optimize_conf(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    positions = np_to_mm(conf.GetPositions())
    simulator.context.setPositions(positions)
    simulator.context.setVelocitiesToTemperature(300 * u.kelvin)
    simulator.minimizeEnergy(maxIterations=500)

    # CHARMM returns all of its positions in nm, so we have to convert back to Angstroms for RDKit
    optimized_positions_nm = simulator.context.getState(getPositions=True).getPositions()
    optimized_positions = optimized_positions_nm.in_units_of(u.angstrom) # match RDKit/MMFF convention

    for i, pos in enumerate(optimized_positions):
        conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))

def get_conformer_energy(mol: Chem.Mol, confId: int = None):
    if confId is None:
        confId = mol.GetNumConformers() - 1
    conf = mol.GetConformer(confId)

    positions = np_to_mm(conf.GetPositions())
    simulator.context.setPositions(positions)
    simulator.context.setVelocitiesToTemperature(300 * u.kelvin)
    energy_kj = simulator.context.getState(getEnergy=True).getPotentialEnergy()
    energy_kcal = energy_kj.in_units_of(u.kilocalories_per_mole) # match RDKit/MMFF convention
    return energy_kcal._value

def get_conformer_energies(mol: Chem.Mol) -> List[float]:
    """Returns a list of energies for each conformer in `mol`.
    """
    energies = []
    Chem.MMFFSanitizeMolecule(mol)
    for conf in mol.GetConformers():
        energy = get_conformer_energy(mol, conf.GetId())
        energies.append(energy)
    
    return np.asarray(energies, dtype=float)

def prune_last_conformer(mol: Chem.Mol, tfd_thresh: float, energies: List[float]) -> Tuple[Chem.Mol, List[float]]:
    """Prunes the last conformer of the molecule.

    If no conformers in `mol` have a TFD (Torsional Fingerprint Deviation) with the last conformer of less than `tfd_thresh`,
    the last conformer is kept. Otherwise, the lowest energy conformer with TFD less than `tfd_thresh` is kept and all other conformers
    are discarded.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned. The conformers in the molecule should be ordered by ascending energy.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.
    energies : list of float
        A list of all the energies of the conformers in `mol`.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning, with conformers sorted by ascending energy.
    energies : list of float
        A list of all the energies of the conformers in `mol` after pruning and sorting by ascending energy.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol, energies

    idx = bisect.bisect(energies[:-1], energies[-1])
    tfd = TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        energies = energies[:-1]
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, energies
    else:
        keep = list(range(0, idx))
        keep.append(mol.GetNumConformers() - 1)
        keep += [x for x in range(idx, mol.GetNumConformers() - 1) if tfd[x] >= tfd_thresh]

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        for i in keep:
            conf = mol.GetConformer(i)
            new.AddConformer(conf, assignId=True)

        return new, [energies[i] for i in keep]

def prune_conformers(mol: Chem.Mol, tfd_thresh: float) -> Chem.Mol:
    """Prunes all the conformers in the molecule.

    Removes conformers that have a TFD (torsional fingerprint deviation) lower than
    `tfd_thresh` with other conformers. Lowest energy conformers are kept.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = get_conformer_energies(mol)
    tfd = tfd_matrix(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the tfd threshold
        if np.all(this_tfd >= tfd_thresh):
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

def calculate_normalizers(mol: Chem.Mol, num_confs: int = 200, pruning_thresh: float = 0.05) -> Tuple[float, float]:
    """Calculates the :math:`E_0` and :math:`Z_0` normalizing constants for a molecule used in the TorsionNet [1]_ paper.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule of interest.
    num_confs : int
        The number of conformers to generate when calculating the constants. Should equal
        the number of steps for each episode of the environment containing this molecule.
    pruning_thresh : float
        TFD threshold for pruning the conformers of `mol`.

    References
    ----------
    .. [1] `TorsionNet paper <https://arxiv.org/abs/2006.07078>`_
    """
    Chem.MMFFSanitizeMolecule(mol)
    confslist = Chem.EmbedMultipleConfs(mol, numConfs = num_confs, useRandomCoords=True)
    if (len(confslist) < 1):
        raise Exception('Unable to embed molecule with conformer using rdkit')
    
    for conf_id in range(mol.GetNumConformers()):
        optimize_conf(mol, conf_id)

    mol = prune_conformers(mol, pruning_thresh)
    energys = get_conformer_energies(mol)
    E0 = energys.min()
    Z0 = np.sum(np.exp(-(energys - E0)))

    mol.RemoveAllConformers()

    return E0, Z0