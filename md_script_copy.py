# testing MD simulations. see md.py
import lmdb
import pickle
import pandas as pd

from ase import Atoms
from ase.build import bulk
from ase.io import Trajectory
from pathlib import Path

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD

def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers.cpu().detach().numpy(), 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms

env = lmdb.open('/data/shared/MLFF/SPICE/maceoff_split/test/data.lmdb',
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
)

df = pd.read_csv('/home/christine/mdsim/md_scripts/spice_molecules_set.csv')
dsets = df['dataset']
init_idxs = df['init_idx']

model_name = 'MACE-MP(M)'

with env.begin() as txn:
    # loop through molecules from molecules_set.csv
    for i in range(len(init_idxs)):
        idx = init_idxs[i]
        value = txn.get(str(idx).encode('ascii'))  # Encode the integer key as bytes
        data = pickle.loads(value)
        atoms = data_to_atoms(data)
        model = MLIPEnum[model_name]
        print(f"Running {model_name} on {idx}")
        result = MD.fn(
            atoms,
            calculator_name=model.name,
            calculator_kwargs={},
            ensemble="nve",
            dynamics="velocityverlet",
            time_step=0.5,
            total_time=25000,
            temperature=300.0,
            zero_linear_momentum=False,
            zero_angular_momentum=False,
            traj_file=f'{model_name}_sims/{dsets[i]}_{idx}.traj',
            traj_interval=100,
        )
