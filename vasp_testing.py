from ase import Atoms
from ase.db import connect
from ase.io import Trajectory
from ase.io import read
import numpy as np
import time
import os

from pymatgen.core import Composition
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

compatibility = MaterialsProject2020Compatibility(check_potcar=False)

atoms = read('/data/shared/MPTrj/original/mp-1185346.extxyz')
entry = ComputedEntry(
    composition=Composition(atoms.get_chemical_formula(empirical=True)),
    energy=atoms.get_total_energy(),
    parameters={
        "run_type": "GGA",  # Manually specify run type as GGA or GGA+U
        "potcar_symbols": ["PBE Li_sv", "PBE Ce", "PBE Zn"],
        "hubbards": {"Li": 0.0, "Ce": 0.0, "Zn": 0.0}
    }
)

adjustments = compatibility.get_adjustments(entry)
print(adjustments)
