import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
import argparse
import numpy as np

from pynamod.Nucleotides_parser import get_all_nucleotides,get_base_ref_frame
from pynamod.Pairs_parser import get_pairs, get_pairs_and_step_params

def parse_pairs_list(nucl_df,pairs_list):
    pairs_df = pd.DataFrame(columns=['restype1','resid1','segid1','resname1','R_frame1','origin1','stand_sel1','exp_sel1',
                             'restype2','resid2','segid2','resname2','R_frame2','origin2','stand_sel2','exp_sel2'])
    for pair in pairs_list:
        nucl1 = nucl_df[nucl_df['resid'] == pair[1]]
        nucl1 = nucl1[nucl1['segid'] == pair[0]]
        nucl2 = nucl_df[nucl_df['resid'] == pair[3]]
        nucl2 = nucl2[nucl2['segid'] == pair[2]]
        nucl1 = nucl1.rename(columns={col_name:col_name+'1' for col_name in nucl_df.columns})
        nucl2 = nucl2.rename(columns={col_name:col_name+'2' for col_name in nucl_df.columns})
        pairs_df = pairs_df.append(pd.concat([nucl1.reset_index(drop=True),nucl2.reset_index(drop=True)],axis=1))
        
    return pairs_df.reset_index(drop=True)

def analyze_structure(mdUniverse=None,file=None,pdb_id=None,leading_strands=[],pairs_list=[],save_ori_and_r = False):
    '''
Full analysis of dna in pdb structure. The function is built to be similar to 3dna algorithm(http://nar.oxfordjournals.org/content/31/17/5108.full).
-----
input:
    mdaUniverse, file, pypdb_id - PDB structure as a mda Universe object, file path or pdb_id respectively
    leading_strands - strands that will be used to set order of parameters calculations
    pairs_list - list of pairs that will be used instead of classifier algorithm to generate pairs DataFrame. Each element of it should be a tuple of the segid and resid of the first nucleotide in pair and then the segid and resid of the second.
----
returns:
    params_df - pandas DataFrame with calculated intra and inter geometrical parameters and resid, segid and nucleotide type of each nucleotides in pair.
    '''
    nucl_df = get_all_nucleotides(mdUniverse=mdUniverse,file=file,pdb_id=pdb_id,leading_strands=leading_strands)
    if pairs_list:
        pairs_df = parse_pairs_list(nucl_df,pairs_list)
    else:
        pairs_df = get_pairs(nucl_df,leading_strands=leading_strands)
    params_df = get_pairs_and_step_params(pairs_df,save_ori_and_r=save_ori_and_r)
    
    return pairs_df,params_df

def analyze_trajectory(md_traj,leading_strands=[],pairs_list=[]):
    '''
Analysis of DNA in PDB structure with optimization to calculation of parameters in frames of molecular dynamics trajectory.
-----
input:
    md_traj - mda Universe that includes trajectory
    leading_strands - strands that will be used to set order of parameters calculations
    pairs_list - list of pairs that will be used instead of classifier algorithm to generate pairs DataFrame. Each element of it should be a tuple of the segid and resid of the first nucleotide in pair, and then the segid and resid of the second.
----
returns:
    params_df - pandas DataFrame with calculated intra and inter geometrical parameters and resid, segid and nucleotide type of each nucleotide in pair per each frame of trajectory.
    '''
    pairs_df,params_df = analyze_structure(mdUniverse=md_traj,leading_strands=leading_strands,pairs_list=pairs_list)
    params_df['frame'] = [0]*len(params_df)
    
    for time_step in tqdm(md_traj.trajectory[1:]):

        new_r_and_o1 = pairs_df[['exp_sel1','stand_sel1']].apply(lambda x: 
                                                    get_base_ref_frame(*x.to_numpy()),axis=1)
        new_r_and_o2 = pairs_df[['exp_sel2','stand_sel2']].apply(lambda x: 
                                                    get_base_ref_frame(*x.to_numpy()),axis=1)
        columns = ['resid1','segid1','restype1','resid2','segid2','restype2']
        step_df = pd.concat([pairs_df[columns].reset_index(),pd.DataFrame.from_dict(new_r_and_o1.to_dict(),columns=['R_frame1','origin1'],orient='index'),
                            pd.DataFrame.from_dict(new_r_and_o2.to_dict(),columns=['R_frame2','origin2'],orient='index')],axis=1)
        time_step_params = get_pairs_and_step_params(step_df)
        time_step_params['frame'] = [time_step.frame]*len(step_df)
        
        params_df = pd.concat([params_df,time_step_params],axis=0,ignore_index=True)
    return params_df