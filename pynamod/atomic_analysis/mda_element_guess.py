import warnings

import MDAnalysis as mda
from MDAnalysis.guesser.default_guesser import DefaultGuesser

# PDBParser warns when the element column is empty; we assign elements via DefaultGuesser next.
_ELEMENT_MISSING_MSG = 'Element information is missing'


def load_pdb_universe(topology, *coordinates, **kwargs):
    '''
    Call :class:`MDAnalysis.core.universe.Universe` with the PDBParser "element missing"
    warning suppressed for this construction only (MDAnalysis init element guess differs
    from :func:`add_guessed_elements`, so callers should still use that when needed).
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=_ELEMENT_MISSING_MSG,
            category=UserWarning,
        )
        return mda.Universe(topology, *coordinates, **kwargs)


def add_guessed_elements(universe):
    '''
    Add ``elements`` topology attribute using MDAnalysis :class:`DefaultGuesser`
    (replaces deprecated :func:`MDAnalysis.topology.guessers.guess_atom_element`).
    '''
    guesser = DefaultGuesser(universe)
    universe.add_TopologyAttr(
        'elements',
        [guesser.guess_atom_element(name) for name in universe.atoms.names],
    )
