import io
import networkx as nx
from MDAnalysis.topology.guessers import guess_atom_element
import MDAnalysis as mda
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from pynamod.atomic_analysis.base_structures import nucleotides_pdb
from pynamod.atomic_analysis.structures_storage import Nucleotides_Storage

'''
This module contains functions to analyze given residues in pdb structures to determine if they are nucleotides and their type. A class Nucleotide then represents their data and function get_all_nucleotides runs the full analysis. Analysis is performed with the usage of networkx library to build graphs based on experimental structures amd standard purine and pyrimidine residues of nucleotides structures. Graphs contain nodes with saved types of atom elements and edges that represent bonds based on distance cut off. Nucleotides are then determined by checking if standard graph is subgraph of experimental graph.
'''

def get_base_u(base_type,nucleotides_pdb=nucleotides_pdb):
    '''
    Function that is used to properly open standard mda universe of a nucleotide of a given type.
    
    Attributes:
    
    **base_type** - string, used as a key for dictionary of standard pdb structures.
    
    **nucleotides_pdb** - dictionary of standard pdb structures.
    
    Returns:
    
    **mdaUniverse** with element types.
    '''
    
    base_u = mda.Universe(io.StringIO(nucleotides_pdb[base_type]), format='PDB')
    base_u.add_TopologyAttr('elements', [guess_atom_element(name) for name in base_u.atoms.names])
    return base_u.atoms

def build_graph(mda_structure, d_threshold=1.6):
    '''
    Creates a graph with nodes representing atoms with their element type and edges representing bonds from mda structure.
    
    Attributes:
    
    **mda_structure** - mda universe to create a graph from.
    
    **d_threshold** - a cutt off used to guess bonds between atoms. If the distance between two atoms is smaller than cut off an edge between them will be added to graph.
    
    Returns:
    
    **graph** - created graph.
    '''
    coords = mda_structure.positions
    dist_mat = cdist(coords, coords)

    dist_mat[dist_mat > d_threshold] = 0
    graph = nx.from_numpy_array(dist_mat)
    nodes_names = {node: {'el': atom.element, 'atom': atom} for node, atom in
                   zip(list(graph.nodes.keys()), mda_structure.atoms)}
    nx.set_node_attributes(graph, nodes_names)
    return graph
    
#Geometrical parameters are calculated based only on atoms of purine or pyrimidine ring, all other atoms should be excluded from analysis
atoms_to_exclude = {'A': [5], 'T': [2, 5, 8], 'G': [5, 8], 'C': [2, 5], 'U': []}


def _check_atom_name(node1, node2):
    '''
    Supporting function for nx.algorithms.isomorphism.ISMAGS to match elements names in nodes.
    '''
    return node1['el'] == node2['el']

def get_base_ref_frame(s_res,e_res):
    '''
    Calculate R frame and origin with the same algorithm as in 3dna.
    
    Attributes:
    
    **e_res, s_res** - mda.atoms of experimental and standard residues with correctly ordered atoms.
    
    
    returns:
    
    **R**, **o** - torch.Tensor R frame and origin of nucleotide with shapes (3,3) and (1,3).
    '''
    s_coord = s_res.positions
    e_coord = e_res.positions

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
    R = torch.DoubleTensor([[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                       [2 * (q2 * q1 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
                       [2 * (q3 * q1 - q0 * q2), 2 * (q3 * q2 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])
    o = torch.DoubleTensor(e_ave - (s_ave*R).sum(axis=1))
    return R,o


nucleotide_graphs = {}
base_graphs = {}
for base in ['A', 'T', 'G', 'C', 'U']:
    mda_str = get_base_u(base)
    nucleotide_graphs[base] = build_graph(mda_str)
    base_graphs[base] = build_graph(mda_str[11:])

atoms_to_exclude = {'A': [5], 'T': [2, 5, 8], 'G': [5, 8], 'C': [2, 5], 'U': []}


def check_if_nucleotide(residue, base_graphs=base_graphs,candidates = ['G', 'T', 'A', 'C', 'U'], use_full_nucleotide=False):
    # TODO: tune speed
    '''
    Finds if atoms of a given residue is nucleotide and gets it type. This function constructs graph of an experimental residue and determines if standard graph is subgraph of it. Then atoms that are not needed in further analysis are removed.
    
    Attributes:
    
    **residue** - mda.atoms of a residue to check.
    
    **base_graphs** - dictionary of graphs that represent 5 standard nucleotides.
    
    **candidates** - list of possible types of nucleotides. Their order matters, as graphs of some standard structures are subgraphs of other standard graphs. 
    
    Returns:
    
    Note that all lists will be empty if residue is not a nucleotide.
    
    **exp_sel** - list of correctly ordered mda atoms of residue.
    
    **stand_sel** - list of correctly ordered mda atoms of a standard nucleotide.
    
    **true_base** - name of this residue in one letter code.

    **use_full_nucleotide** - bool, whether to use a full nucleotide for generating output graphs instead of the acidic bases only
    '''
    stand_sel = []
    exp_sel = []
    true_base = ''
    graph = build_graph(residue)
    for base in candidates:
        if use_full_nucleotide:
            base_graph = nucleotide_graphs[base].copy()
        else:
            base_graph = base_graphs[base].copy()
        ismags_inst = nx.algorithms.isomorphism.ISMAGS(graph, base_graph, node_match=_check_atom_name)
        mapping = list(ismags_inst.find_isomorphisms(symmetry=True))

        if mapping != []:
            # надо проверять, что в меппинге хватает атомов, надо, чтобы не было лишних атомов
            mapping = dict(zip(mapping[0].values(), mapping[0].keys()))

            true_base = base
            if not use_full_nucleotide:
                for i in atoms_to_exclude[true_base]:
                    del (mapping[i])

            for id_sub, id_mol in sorted(mapping.items()):
                exp_sel.append(ismags_inst.graph.nodes[id_mol]['atom'])
                stand_sel.append(ismags_inst.subgraph.nodes[id_sub]['atom'])

            break
    return exp_sel, stand_sel, true_base




class Nucleotide:
    '''
    Represents data of a single nucleotide. It exists as a reference to a nucleotide storage object which contains data of all nucleotides in a structures. All of its main variables are properties that get information from the storage class or set it there. These variables include:
    
    **resid**, **segid** - nucleotide data from MDAnalysis.
    
    **restype** - guessed priorly by check_if_nucleotide function.
    
    **leading_strands** - bool, leading strands in strucutre are provided by user when analysis of a structure starts.
    
    **origin**, **ref_frame** - torch.tensors of shapes (1,3) and (3,3). If these variables are used and storage contains None values, get_base_ref_frame function will be called to calculate them.
    
    **s_residue**, **e_residue** - mda.atoms of standard and experimental residues. If these variables are used and storage contains None values, s_residue will be set from standrad pdb file and e_resiue will either be selected from mda Universe stored in this nucleotide coarse-grained structure class instance. If coarse-grained structure does not have mda Universe, e_residue will also be set from standard pdb file.
    
    **next_nucleotide**, **previous_nucleotide** - Nucleotide class links to adjacent nucleotides. The order for both leading and lagging chain is considered to be 5' end to 3'.
    '''
    def __init__(self,storage_class, ind):
        
        self.storage_class = storage_class
        self.ind = ind


    def __lt__(self, other):
        #Sorted nucleotides start with those that are in leading strands, inside leading and lagging strands they are ordered by increase of resid.
        if self.leading_strand != other.leading_strand:
            return self.leading_strand > other.leading_strand
        else:
            return self.resid < other.resid
        
    def __eq__(self,other):
        return self.storage_class == other.storage_class and self.ind == other.ind
    
    
    def __setter(self,attr,value):
        getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind] = value
        
    def __getter(self,attr):
        return getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind]
        
    def __set_property(attr):
        setter = lambda self,value: self.__setter(value,attr=attr)
        getter = lambda self: self.__getter(attr=attr)
        return property(fset=setter,fget=getter)
        
    restype = __set_property('restype')
    resid = __set_property('resid')
    segid = __set_property('segid')
    leading_strand = __set_property('leading_strand')

    
    @property
    def origin(self):
        value = self.__getter('origin')
        if value is None:
            R,o = get_base_ref_frame(self.s_residue,self.e_residue)
            self.__setter('ref_frame',R)
            self.__setter('origin',o)
            value = o
        return value
    
    @origin.setter
    def origin(self,value):
        self.__setter('origin',value)
        
    @property
    def ref_frame(self):
        value = self.__getter('ref_frame')
        if value is None:
            R,o = get_base_ref_frame(self.s_residue,self.e_residue)
            self.__setter('ref_frame',R)
            self.__setter('origin',o)
            value = R
        return value
    
    @ref_frame.setter
    def ref_frame(self,value):
        self.__setter('ref_frame',value)
        
    
        
    @property
    def s_residue(self):
        value = self.__getter('s_residue')
        if value is None:
            value = get_base_u(self.restype)
            self.__setter('s_residue',value)
        return value
    
    @s_residue.setter
    def s_residue(self,value):
        self.__setter('s_residue',value)
        
        
    @property
    def e_residue(self):
        value = self.__getter('e_residue')
        if value is None:
            if self.storage_class.mda_u is not None:
                u = self.storage_class.mda_u.select_atoms(f'resid {self.resid} and segid {self.segid}')
            else:
                u = get_base_u(self.restype)
                
            exp_sel, stand_sel, _ = check_if_nucleotide(residue,candidates=[self.restype])
            self.__setter('s_residue',sum(stand_sel))
            self.__setter('e_residue',sum(exp_sel))
            value = exp_sel
    
        return value
    
    @e_residue.setter
    def e_residue(self,value):
        self._setter('e_residue',value)
        
        
    @property
    def next_nucleotide(self):
        ind = self.storage_class.e_residues.index(self.e_residue) + 1
        if ind == len(self.storage_class) or self.storage_class.leading_strands[ind] != self.leading_strand:
            return None

        return self.storage_class[ind]

    @property
    def previous_nucleotide(self):
        ind = self.storage_class.e_residues.index(self.e_residue) - 1
        if ind == -1 or self.storage_class.leading_strands[ind] != self.leading_strand:
            return None

        return self.storage_class[ind]

    def __repr__(self):
        return f'<Nucleotide with type {self.restype}, resid {self.resid} and segid {self.segid}>'
    

        
def get_all_nucleotides(DNA_Structure,leading_strands,sel,use_full_nucleotide=False):
    '''
    Applies check_if_nucleotide function to each residue in selection from mda Universe stored in DNA_Structure. All atoms with altLocs are ignored.
    
    Attributes:

    **DNA_structure** - object with information to ananlyze.
    
    **leading_strands** - list of leading strands segids in the given structure.
    
    **sel** - selection that is applied to mda Universe before analysis. Standard residues only contain C, O and N atoms, therefore it is better to only use these atoms in experimental structure for efficiency reasons.

    **use_full_nucleotide** - bool, whether to use a full nucleotide for generating output graphs instead of the acidic bases only.
    
    Returns:
    
    **nucleotides_data** - Nucleotides_Storage object that contains information about all found nucleotides.
    '''
    nucleotides_data = Nucleotides_Storage(Nucleotide,DNA_Structure.u)
    if sel:
        sel = DNA_Structure.u.select_atoms(sel)
    else:
        sel = DNA_Structure.u.select_atoms('(type C or type O or type N) and not protein')
    sel = sel[sel.altLocs == '']
    for res_numb, residue in enumerate(sel.residues):
        residue_str = residue.atoms
        if 10 < len(residue_str) < 40:  # FIXME
            exp_sel, stand_sel, base = check_if_nucleotide(residue_str, use_full_nucleotide=use_full_nucleotide)
            if base != '':
                leading_strand = residue.segid in leading_strands
                R,o = get_base_ref_frame(sum(stand_sel),sum(exp_sel))
                nucleotides_data.append(base, residue.resid, residue.segid, leading_strand,R,o.reshape(1,3),sum(stand_sel),sum(exp_sel),None)
                
    if len(nucleotides_data) == 0:
        raise ValueError('No nucleotides found.')
        
    nucleotides_data.sort('leading_strand','resid')
    nucleotides_data = nucleotides_data[nucleotides_data.leading_strands] + nucleotides_data[[not i for i in nucleotides_data.leading_strands]]
    return nucleotides_data
