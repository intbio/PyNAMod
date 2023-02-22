import networkx as nx
from MDAnalysis.topology.guessers import guess_atom_element
import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pypdb

def build_graph(mda_structure, d_threshold=1.6):
    '''
    Create a graph from mda structure.
    -----
    mda_structure - a structure to create a graph from
    graph - created graph
    '''
    coords = mda_structure.positions
    dist_mat = cdist(coords,coords)
    
    dist_mat[dist_mat > d_threshold] = 0
    graph = nx.from_numpy_matrix(dist_mat)
    nodes_names = {node : {'el':atom.element,'atom':atom} for node,atom in zip(list(graph.nodes.keys()),mda_structure.atoms)}
    nx.set_node_attributes(graph,nodes_names)
    return graph    

def _check_atom_name(node1,node2):
    '''
    Supporting function for nx.algorithms.isomorphism.ISMAGS to match elements names in nodes.
    '''
    return node1['el'] == node2['el']

base_graphs = {}
for base in ['A','T','G','C','U']:
    mda_str = mda.Universe(f'/home/_shared/package_dev/PyNAMod/pynamod/Base_structures/Atomic_{base}.pdb')
    mda_str.add_TopologyAttr('elements',[guess_atom_element(name) for name in mda_str.atoms.names])
    base_graphs[base] = build_graph(mda_str.select_atoms('not name ORI'))
atoms_to_exclude = {'A':[5],'T':[2,5,8],'G':[5,8],'C':[2,5],'U':[]}

def check_if_nucleotide(residue,base_graphs=base_graphs):
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
    for base in ['G','T','A','C','U']:
        base_graph = base_graphs[base].copy()
        ismags_inst = nx.algorithms.isomorphism.ISMAGS(graph,base_graph,node_match=_check_atom_name)
        mapping = list(ismags_inst.find_isomorphisms(symmetry=True))
        
        del(base_graph)

        if mapping != []:
            # надо проверять, что в меппинге хватает атомов, надо, чтобы не было лишних атомов
            mapping = dict(zip(mapping[0].values(),mapping[0].keys()))

            true_base = base
            for i in atoms_to_exclude[true_base]:
                del(mapping[i])

            for id_sub,id_mol in sorted(mapping.items()):

                exp_sel += [ismags_inst.graph.nodes[id_mol]['atom']]
                stand_sel += [ismags_inst.subgraph.nodes[id_sub]['atom']]
            
            break
    return exp_sel,stand_sel,true_base

def get_base_ref_frame(e_res,s_res):
    '''
    Calculate R frame and origin with the same algorithm as in 3dna.
    -----
    e_res,s_res - experimental residue, standard residue(both are mda.atoms type) with correctly ordered atoms(they can be got by check if nucleotide function)
    returns R frame and origin of nucleotide
    '''
    s_coord = s_res.positions
    e_coord = e_res.positions
    
    
    s_ave = np.mean(s_coord,axis=0)
    e_ave = np.mean(e_coord,axis=0)
    
    
    N = len(e_coord)
    i = np.ones((N,1))
    cov_mat = (s_coord.T.dot(e_coord) - s_coord.T.dot(i).dot(i.T).dot(e_coord)/N)/(N-1)
    
    
    M = np.array([[cov_mat[0,0]+cov_mat[1,1]+cov_mat[2,2],cov_mat[1,2]-cov_mat[2,1],cov_mat[2,0]-cov_mat[0,2],cov_mat[0,1]-cov_mat[1,0]],
             [cov_mat[1,2]-cov_mat[2,1],cov_mat[0,0]-cov_mat[1,1]-cov_mat[2,2],cov_mat[0,1]+cov_mat[1,0],cov_mat[2,0]+cov_mat[0,2]],
             [cov_mat[2,0]-cov_mat[0,2],cov_mat[0,1]+cov_mat[1,0],-cov_mat[0,0]+cov_mat[1,1]-cov_mat[2,2],cov_mat[1,2]+cov_mat[2,1]],
             [cov_mat[0,1]-cov_mat[1,0],cov_mat[2,0]+cov_mat[0,2],cov_mat[1,2]+cov_mat[2,1],-cov_mat[0,0]-cov_mat[1,1]+cov_mat[2,2]]])
    eigen = np.linalg.eig(M)
    index = np.argmax(eigen[0])
    q = eigen[1][:,index]
    #q *= -1
    
    q0,q1,q2,q3 = q
    R = np.array([[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
              [2*(q2*q1+q0*q3),q0*q0-q1*q1+q2*q2-q3*q3,2*(q2*q3-q0*q1)],
              [2*(q3*q1-q0*q2),2*(q3*q2+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3]])
    o = e_ave - s_ave.dot(R.T)
    return R,o


def get_all_nucleotides(mdUniverse=None,file=None,pdb_id=None,leading_strands=[]):
    '''
    Create data frame with data about nucleotides from a given pdb.
    -----
    pdb - structure to parse
    dirpath - path to directory with file
    extraction_with_io - if set to True, function will get structure with pypdb.get_pdb_file instead of reading pdb file.
    base_graphs - dictionary of graphs which represent 5 nucleotides
    nucleotides_df - result data frame
    '''
    nucleotides_dict = {}
    
    if mdUniverse:
        u = mdUniverse
    elif pdb_id:
        u = mda.Universe(io.StringIO(pypdb.get_pdb_file(pdb_id)), format='PDB')
    elif file:
        u = mda.Universe(filename)
    else:
        return None
    if not hasattr(u.atoms,'elements'):
        u.add_TopologyAttr('elements',[guess_atom_element(name) for name in u.atoms.names])
    
    sel = u.select_atoms('(type C or type O or type N) and not protein')
    sel = sel[sel.altLocs=='']
    for res_numb,residue in enumerate(sel.residues):
        residue_str = residue.atoms
        if 10 < len(residue_str) < 40: # FIXME
            exp_sel,stand_sel,base = check_if_nucleotide(residue_str)
            if base != '':
                R,o = get_base_ref_frame(sum(exp_sel),sum(stand_sel))
                nucleotides_dict[res_numb] = [base,residue.resid,residue.segid,residue.resname,R,o,sum(stand_sel),sum(exp_sel)]
    nucleotides_df = pd.DataFrame.from_dict(nucleotides_dict,columns=
                                            ['restype','resid','segid','resname','R_frame','origin','stand_sel','exp_sel'],orient='index')
    
    corrected_df = pd.DataFrame(columns=nucleotides_df.columns)
    if leading_strands:
        for name in leading_strands:
            strand_nucleotides = nucleotides_df[nucleotides_df['segid']==name].sort_values('resid')
            corrected_df = pd.concat((corrected_df,strand_nucleotides))
        strand_nucleotides = nucleotides_df.query(f"segid not in '{''.join(leading_strands)}'").reset_index(drop=True)
        corrected_df = pd.concat((corrected_df,strand_nucleotides))
        return corrected_df.reset_index(drop=True)
    else:
        return nucleotides_df
