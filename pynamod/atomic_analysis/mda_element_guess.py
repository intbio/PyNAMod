from MDAnalysis.guesser.default_guesser import DefaultGuesser


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
