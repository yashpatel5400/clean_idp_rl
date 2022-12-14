import time
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import json
import tqdm
import numpy as np

from main.utils import *

def create_t_alkane(i):
    cin = 'C' * i
    m = Chem.MolFromSmiles(f'CCCC({cin})CCCC')
    m = Chem.rdmolops.AddHs(m)

    AllChem.EmbedMultipleConfs(m, numConfs=2000, numThreads=-1)
    md_sim = MDSimulator(m)
    md_sim.optimize_confs(m, numThreads=-1)

    energys = md_sim.get_conformer_energies(m)
    argsorted = np.argsort(energys)
    print(len(TorsionFingerprints.CalculateTorsionLists(m)[0]))
    standard = energys.min()
    total = np.sum(np.exp(-(energys-standard)))

    Chem.MolToMolFile(m, f'transfer_test_t_chain/{i}.mol', confId=int(argsorted[0]))

    out = {
        'mol': Chem.MolToSmiles(m, isomericSmiles=False),
        'standard': standard,
        'total': total
    }

    return out

for i in range(4,14):
    out = create_t_alkane(i)
    with open(f'transfer_test_t_chain/{i}.json', 'w') as fp:
        json.dump(out, fp)
