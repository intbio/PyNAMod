'''Tests that standard DNA/RNA bases are recognized by nucleotide graph matching.'''

import io
import sys
import unittest
from pathlib import Path

# Load atomic_analysis without importing pynamod.__init__ (heavy scipy/torch stack).
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from atomic_analysis.base_structures import nucleotides_pdb
from atomic_analysis.mda_element_guess import add_guessed_elements, load_pdb_universe
from atomic_analysis.nucleotides_parser import check_if_nucleotide


class TestNucleotideParsing(unittest.TestCase):
    '''Same atom selection as DNA_Structure.build_from_u / get_all_nucleotides.'''

    sel = '(type C or type O or type N) and not protein'

    def test_all_standard_bases_recognized_with_cno_selection(self):
        '''A, T, G, C cover DNA; U covers RNA; none must be skipped by length or matcher.'''
        for base in ('A', 'T', 'G', 'C', 'U'):
            with self.subTest(base=base):
                u = load_pdb_universe(io.StringIO(nucleotides_pdb[base]), format='PDB')
                add_guessed_elements(u)
                residue_atoms = u.residues[0].atoms.select_atoms(self.sel)
                n = len(residue_atoms)
                self.assertGreater(n, 10, f'{base}: too few C/N/O atoms after selection')
                self.assertLess(n, 40, f'{base}: unexpected residue size; filter may skip it')
                exp_sel, stand_sel, found = check_if_nucleotide(
                    residue_atoms, use_full_nucleotide=False
                )
                self.assertEqual(found, base, f'{base}: wrong or empty type from check_if_nucleotide')
                self.assertTrue(exp_sel, f'{base}: empty experimental atom list')
                self.assertTrue(stand_sel, f'{base}: empty standard atom list')
                self.assertEqual(len(exp_sel), len(stand_sel), f'{base}: mapping length mismatch')

    def test_dna_sequence_has_four_types_after_build(self):
        try:
            import pynamod
        except ImportError as err:
            self.skipTest(f'full pynamod stack not available: {err}')
        cg = pynamod.CG_Structure()
        cg.build_dna(sequence='ATCG')
        types = set(cg.dna.nucleotides.restypes)
        self.assertEqual(types, {'A', 'T', 'C', 'G'})


if __name__ == '__main__':
    unittest.main()
