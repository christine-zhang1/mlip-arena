import numpy as np
import os

from ase import units
from ase.io import read
from ase.db import connect
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from pymatgen.io.vasp import Kpoints
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun

from jobflow import run_locally

from atomate2.vasp.jobs.mp import MPGGAStaticMaker
from atomate2.vasp.sets.mp import MPGGAStaticSetGenerator

db_name = 'molecules_CHGNet.db' # replace name
db_name_id = db_name.split('.')[0]
db = connect(db_name)
num_molecules = db.count()
descriptions = []

for row in db.select():
    atoms = row.toatoms()
    desc = row['description']
    descriptions.append(desc)

    atoms.center(vacuum=5)
    
    structure = AseAtomsAdaptor.get_structure(atoms)

    input_set_generator = MPGGAStaticSetGenerator(
        user_incar_settings=dict(
            # molecule
            # ISYM=0,
            ISMEAR=0, 
            SIGMA=0.03,
            ENCUT=2000,
            # gpu
            KPAR=1,
            NSIM=16,
            # dispersion correction for image
            LDIPOL = True,
            IDIPOL = 4,
        ),
        user_kpoints_settings=Kpoints(),
        sort_structure=True, #False
    )
    
    # wasn't sure if run_locally supports writing folders to subdirectories (not the cwd)
    # so I'm changing into a subdirectory here
    output_dir = f"./{db_name_id}"
    os.makedirs(output_dir, exist_ok=False)
    os.chdir(output_dir)

    flow = MPGGAStaticMaker(input_set_generator=input_set_generator).make(structure)
    run_locally(flow, create_folders=True)
    
# change back to original directory, and read through all the jobs that were created in output_dir
os.chdir('..')
# since the jobs are named by timestamp, they should be in order of the molecules in the database
files = [f for f in sorted(os.listdir(output_dir))]
assert(len(descriptions) == len(files))

for i in range(len(files)):
    desc = descriptions[i]
    job = files[i]
    
    out_db = connect(f'mpgga_{db_name_id}.db')
    vrun = Vasprun(os.path.join(output_dir, job, 'vasprun.xml.gz'))

    step = vrun.ionic_steps[-1]
    atoms = step['structure'].to_ase_atoms()

    atoms.calc = SinglePointCalculator(
        atoms=atoms, 
        energy=step['e_0_energy'],
        forces=step['forces'],
        stress=np.array(step['stress']) * -0.1 * units.GPa
    )

    out_db.write(atoms, key_value_pairs={'description': desc})
        