import networkx as nx
from MDAnalysis.topology.guessers import guess_atom_element
import MDAnalysis as mda
import numpy as np
import torch
from scipy.spatial.distance import cdist
from more_itertools import pairwise
import io

from pynamod.atomic_analysis.base_structures import nucleotides_pdb

class Nucleotide():
    def __init__(self, restype, resid, segid, in_leading_strand,R=None,o=None,s_res=None, e_res=None):
        self.restype = restype
        self.resid = resid
        self.segid = segid
        self.in_leading_strand = in_leading_strand
        self.previous_nucleotide = None
        self.next_nucleotide = None
        self.base_pair = None
        self.s_res = s_res
        self.e_res = e_res
        if R is not None:
            self.R = R
        elif s_res is not None:
            self.get_base_ref_frame()
        else:
            self.R = torch.eye(3)
            self.o = torch.zeros(1,3)
        if o is not None:
            self.o = o


    def __lt__(self, other):
        if self.in_leading_strand != other.in_leading_strand:
            return self.in_leading_strand > other.in_leading_strand
        else:
            return self.resid < other.resid
    
        
    def copy(self):
        return Nucleotide(self.restype, self.resid, self.segid, self.in_leading_strand,self.R.clone(),self.o.clone(), self.s_res, self.e_res)

    def get_base_ref_frame(self):
        '''
        Calculate R frame and origin with the same algorithm as in 3dna.
        -----
        e_res,s_res - experimental residue, standard residue(both are mda.atoms type) with correctly ordered atoms(they can be got by check if nucleotide function)
        returns R frame and origin of nucleotide
        '''
        s_coord = self.s_res.positions
        e_coord = self.e_res.positions

        s_ave = torch.from_numpy(np.mean(s_coord, axis=0))
        e_ave = torch.from_numpy(np.mean(e_coord, axis=0))

        N = len(e_coord)
        i = np.ones((N, 1))
        cov_mat = (s_coord.T.dot(e_coord) - s_coord.T.dot(i).dot(i.T).dot(e_coord) / N) / (N - 1)

        M = np.array([[cov_mat[0, 0] + cov_mat[1, 1] + cov_mat[2, 2], cov_mat[1, 2] - cov_mat[2, 1],
                       cov_mat[2, 0] - cov_mat[0, 2], cov_mat[0, 1] - cov_mat[1, 0]],
                      [cov_mat[1, 2] - cov_mat[2, 1], cov_mat[0, 0] - cov_mat[1, 1] - cov_mat[2, 2],
                       cov_mat[0, 1] + cov_mat[1, 0], cov_mat[2, 0] + cov_mat[0, 2]],
                      [cov_mat[2, 0] - cov_mat[0, 2], cov_mat[0, 1] + cov_mat[1, 0],
                       -cov_mat[0, 0] + cov_mat[1, 1] - cov_mat[2, 2], cov_mat[1, 2] + cov_mat[2, 1]],
                      [cov_mat[0, 1] - cov_mat[1, 0], cov_mat[2, 0] + cov_mat[0, 2], cov_mat[1, 2] + cov_mat[2, 1],
                       -cov_mat[0, 0] - cov_mat[1, 1] + cov_mat[2, 2]]])
        eigen = np.linalg.eig(M)
        index = np.argmax(eigen[0])
        q = eigen[1][:, index]
        # q *= -1

        q0, q1, q2, q3 = q
        self.R = torch.DoubleTensor([[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                           [2 * (q2 * q1 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                           [2 * (q3 * q1 - q0 * q2), 2 * (q3 * q2 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])
        self.o = torch.DoubleTensor(e_ave - (s_ave*self.R).sum(axis=1))
    



def build_graph(mda_structure, d_threshold=1.6):
    '''
    Create a graph from mda structure.
    -----
    mda_structure - a structure to create a graph from
    graph - created graph
    '''
    coords = mda_structure.positions
    dist_mat = cdist(coords, coords)

    dist_mat[dist_mat > d_threshold] = 0
    graph = nx.from_numpy_array(dist_mat)
    nodes_names = {node: {'el': atom.element, 'atom': atom} for node, atom in
                   zip(list(graph.nodes.keys()), mda_structure.atoms)}
    nx.set_node_attributes(graph, nodes_names)
    return graph


def _check_atom_name(node1, node2):
    '''
    Supporting function for nx.algorithms.isomorphism.ISMAGS to match elements names in nodes.
    '''
    return node1['el'] == node2['el']


base_graphs = {}
for base in ['A', 'T', 'G', 'C', 'U']:
    mda_str = mda.Universe(io.StringIO(nucleotides_pdb[base]), format='PDB')
    mda_str.add_TopologyAttr('elements', [guess_atom_element(name) for name in mda_str.atoms.names])
    base_graphs[base] = build_graph(mda_str.select_atoms('not name ORI'))
atoms_to_exclude = {'A': [5], 'T': [2, 5, 8], 'G': [5, 8], 'C': [2, 5], 'U': []}


def check_if_nucleotide(residue, base_graphs=base_graphs):
    # tune speed
    '''
    Find if residue of pdb structure is nucleotide and get it type.
    -----
    residue - residue(must be mda.atoms type) to check.
    base_graphs - dictionary of graphs which represent 5 nucleotides
    Returns empty lists if residue is not a nucleotide. If it is:
    exp_sel - list of correctly ordered mda atoms of residue
    stand_sel - list of correctly ordered mda atoms of a reference nucleotide. These two lists are necessary to calculate R frame.
    true_base - name of this residue in one letter code
    '''

    stand_sel = []
    exp_sel = []
    true_base = ''
    graph = build_graph(residue)
    for base in ['G', 'T', 'A', 'C', 'U']:
        base_graph = base_graphs[base].copy()
        ismags_inst = nx.algorithms.isomorphism.ISMAGS(graph, base_graph, node_match=_check_atom_name)
        mapping = list(ismags_inst.find_isomorphisms(symmetry=True))

        if mapping != []:
            # надо проверять, что в меппинге хватает атомов, надо, чтобы не было лишних атомов
            mapping = dict(zip(mapping[0].values(), mapping[0].keys()))

            true_base = base
            for i in atoms_to_exclude[true_base]:
                del (mapping[i])

            for id_sub, id_mol in sorted(mapping.items()):
                exp_sel.append(ismags_inst.graph.nodes[id_mol]['atom'])
                stand_sel.append(ismags_inst.subgraph.nodes[id_sub]['atom'])

            break
    return exp_sel, stand_sel, true_base


def get_all_nucleotides(DNA_Structure,leading_strands,sel):
    '''
    Create data frame with data about nucleotides from a given pdb.
    -----
    pdb - structure to parse
    dirpath - path to directory with file
    extraction_with_io - if set to True, function will get structure with pypdb.get_pdb_file instead of reading pdb file.
    base_graphs - dictionary of graphs which represent 5 nucleotides
    nucleotides_df - result data frame
    '''
    nucleotides = []
    if sel:
        sel = DNA_Structure.u.select_atoms(sel)
    else:
        sel = DNA_Structure.u.select_atoms('(type C or type O or type N) and not protein')
    sel = sel[sel.altLocs == '']
    for res_numb, residue in enumerate(sel.residues):
        residue_str = residue.atoms
        if 10 < len(residue_str) < 40:  # FIXME
            exp_sel, stand_sel, base = check_if_nucleotide(residue_str)
            if base != '':
                in_leading_strands = residue.segid in leading_strands
                nucl = Nucleotide(base, residue.resid, residue.segid, in_leading_strands,s_res=sum(stand_sel), e_res=sum(exp_sel))
                nucleotides.append(nucl)

    nucleotides.sort()
    for cur_nucleotide,next_nucleotide in pairwise(nucleotides):
        if cur_nucleotide.in_leading_strand == next_nucleotide.in_leading_strand == 1:
            cur_nucleotide.next_nucleotide = next_nucleotide
            next_nucleotide.previous_nucleotide = cur_nucleotide
        elif cur_nucleotide.in_leading_strand == next_nucleotide.in_leading_strand == 0:
            cur_nucleotide.previous_nucleotide = next_nucleotide
            next_nucleotide.next_nucleotide = cur_nucleotide
    return nucleotides
