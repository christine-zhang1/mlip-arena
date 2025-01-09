# testing MD simulations. see md.py
from ase.io import read

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD


model_name = 'CHGNet'

# loop through molecules from mptrj_ids.txt
with open('mptrj_ids_escher.txt', 'r') as f:
    for i, line in enumerate(f):
        atoms = read(line.strip())
        model = MLIPEnum[model_name]
        print(f"Running {model_name} on molecule number {i}")
        molecule_num = int(line.split('.')[0].split('-')[1])
        result = MD.fn(
            atoms,
            calculator_name=model.name,
            calculator_kwargs={},
            ensemble="nve",
            dynamics="velocityverlet",
            time_step=0.5,
            total_time=25000,
            temperature=300.0,
            traj_file=f'CHGNet_MPTrj_sims/{i}_{molecule_num}.traj',
            traj_interval=100,
        )
