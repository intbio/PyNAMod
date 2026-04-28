import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Monkeypatch guess_atom_element to suppress DeprecationWarning (used internally by guess_TopologyAttrs)
import MDAnalysis.topology.guessers as _mda_guessers
_guess_atom_element_orig = _mda_guessers.guess_atom_element
def _guess_atom_element_quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return _guess_atom_element_orig(*args, **kwargs)
_mda_guessers.guess_atom_element = _guess_atom_element_quiet

from pynamod.structures import *
from pynamod.energy.energy import Energy
from pynamod.MC_simulation.iterator import Iterator
import pynamod.geometry