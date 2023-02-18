from scipy.spatial.distance import pdist
import MDAnalysis as mda
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
from pynamod.bp_step_geometry import get_params_for_single_step_stock

#revise
def fix_pairs_df(pairs_df,nucl_df,leading_strands=[]):
    '''
Add missing pairs by context after classifier algorithm.
-----
Input:
    pairs_df - pandas DataFrame with pairs from ckassifier algorithm
    nucl_df - dataframe with nucleotides data
    leading_strands - strands that will be used to set order of parameters calcuctions
-----
Returns:
    pairs_df - corrected data frame with found pairs
    '''
    seg_dict = {segid: sorted(nucl_df.loc[nucl_df['segid']==segid,'resid'].to_numpy()) for segid in set(nucl_df.segid)}
    resids1_array = pairs_df['resid1'].to_numpy()
    for seg_name in set(pairs_df.segid1):
        for resid1 in np.sort(seg_dict[seg_name])[1:-1]:
          # resid_not_in_pairs = pairs_df.query(
           #     f"(resid1 =={resid1} and segid1 == {seg_name})or (resid1 =={resid1} and segid1 == {seg_name})")
            if resid1 not in resids1_array:
                neigbor1_res2_data = pairs_df.loc[pairs_df['resid1']==resid1-1,['resid2','segid2']].reset_index()
                neigbor2_res2_data = pairs_df.loc[pairs_df['resid1']==resid1+1,['resid2','segid2']].reset_index()
                try:
                    neigbor1_resid2,neigbor2_resid2 = neigbor1_res2_data.loc[0,'resid2'],neigbor2_res2_data.loc[0,'resid2']
                except KeyError:
                    continue
                res_consecutive = abs(neigbor1_resid2-neigbor2_resid2) == 2 
                same_seg = neigbor1_res2_data.loc[0,'segid2'] == neigbor2_res2_data.loc[0,'segid2']
                
                if res_consecutive and same_seg:
                    segid2 = neigbor1_res2_data.loc[0,'segid2']
                    resid2 = np.mean((neigbor1_resid2,neigbor2_resid2))
                    
                    resid1_data = nucl_df[nucl_df['resid']==resid1].query(f"segid == '{seg_name}'").reset_index(drop=True)
                    resid2_data = nucl_df[nucl_df['resid']==resid2].query(f"segid == '{segid2}'").reset_index(drop=True)
                    
                    if resid1_data.empty or resid2_data.empty:
                        continue
                    
                    columns_to_get = ['restype','resid','segid','R_frame','origin','stand_sel','exp_sel']
                    nucl1_names = ['restype1','resid1','segid1','R_frame1','origin1','stand_sel1','exp_sel1']
                    nucl1 = resid1_data[columns_to_get].rename(columns=dict(zip(columns_to_get,nucl1_names)))
                    
                    nucl2_names = ['restype2','resid2','segid2','R_frame2','origin2','stand_sel2','exp_sel2']
                    nucl2 = resid2_data[columns_to_get].rename(columns=dict(zip(columns_to_get,nucl2_names)))
                    
                    new_pair = pd.concat([nucl1,nucl2],axis=1)
                    new_pair['pair_name'] = ''.join(sorted(set(new_pair.loc[0,['restype1','restype2']])))
                    pairs_df = pairs_df.append(new_pair,ignore_index=True)
    corrected_df = pd.DataFrame(columns=pairs_df.columns)
    for name in leading_strands:
        strand_pairs = pairs_df[pairs_df['segid1']==name].sort_values('resid1')
        corrected_df = pd.concat((corrected_df,strand_pairs))
    strand_pairs = pairs_df.query(f"segid1 not in '{''.join(leading_strands)}'")
    corrected_df = pd.concat((corrected_df,strand_pairs))
    return pairs_df.reset_index(drop=True)
#review
with open('../pynamod/classifier.pkl','rb') as f:
    classifier = pickle.load(f)
def get_pairs(nucl_df,leading_strands=[]):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algorithm.
    -----
    nucl_df - dataframe with nucleotides data
    Return data frame with found pairs
    '''
    candidates_df_columns = ['restype1','resid1','segid1','resname1','R_frame1','origin1','stand_sel1','exp_sel1',
                             'restype2','resid2','segid2','resname2','R_frame2','origin2','stand_sel2','exp_sel2']

    nucleotides_combinations = np.array(list(combinations(nucl_df[['restype','resid','segid','resname',
                                                                   'R_frame','origin','stand_sel','exp_sel']].to_numpy(),2)))
    pairs_candidates_df = pd.DataFrame(nucleotides_combinations.reshape(-1,len(candidates_df_columns)),
                                       columns=candidates_df_columns)

    pairs_candidates_df['dist'] = pdist(np.stack(nucl_df['origin']))

    pairs_candidates_df = pairs_candidates_df[pairs_candidates_df['dist']<4]
    pairs_candidates_df['pair_name'] = np.sum(pairs_candidates_df[['restype1','restype2']],axis=1)
    pairs_candidates_df['pair_name'] = pairs_candidates_df['pair_name'].apply(lambda x: ''.join(sorted(x)).upper())
    pairs_candidates_df = pairs_candidates_df[pairs_candidates_df.pair_name.apply(lambda x: x in ('AT','CG','AU'))]
    zyx = np.apply_along_axis(lambda x: R.match_vectors(*x)[0].as_euler('zyx', degrees=True),
                                  1,
                                  pairs_candidates_df[['R_frame1','R_frame2']].to_numpy())

    pairs_candidates_df = pd.concat([pairs_candidates_df.reset_index(),
                                     pd.DataFrame(zyx,columns=['z','y','x'])],axis=1)
    pairs_candidates_df.loc[pairs_candidates_df['x'] < 0,'x'] += 360

    pairs = classifier.predict(pairs_candidates_df[['z','y','x','dist']].to_numpy())
    pairs_candidates_df['if_pair'] = pairs.astype(bool)
    pairs_candidates_df = pairs_candidates_df[pairs_candidates_df['if_pair']]
    pairs_candidates_df = fix_pairs_df(pairs_candidates_df[['pair_name']+candidates_df_columns],nucl_df,leading_strands=leading_strands)
    
    return(pairs_candidates_df)

def apply_get_params_to_df(df,ori_r_column_names,param_column_names,pair_params=False):
    series = df[ori_r_column_names].apply(lambda x: get_params_for_single_step_stock(*x,pair_params=pair_params),axis=1)
    param_df = pd.DataFrame.from_dict(series.to_dict(),columns=param_column_names,orient='index')
    
    return param_df

def get_pairs_and_step_params(pairs_df,save_ori_and_r=False):
    pair_params_names = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening']
    step_params_names = ['Shift','Slide','Rise','Tilt','Roll','Twist']
    param_df = apply_get_params_to_df(pairs_df,['origin2','origin1','R_frame2','R_frame1'],pair_params_names+['om','Rm'],pair_params=True)
    param_df = param_df.reset_index(drop=True)
    
    df_len = len(param_df)
    param_df_copy = param_df.loc[:(df_len-2),['om','Rm']].rename(index=dict(zip(range(df_len-1),range(1,df_len))),columns={'om':'om1','Rm':'Rm1'})
    step_df = pd.concat([param_df.loc[1:,['om','Rm']],param_df_copy],axis=1)
    step_df = apply_get_params_to_df(step_df,['om1','om','Rm1','Rm'],step_params_names+['om','Rm'])
    step_df.loc[0,step_params_names] = [0]*6
    if save_ori_and_r:
        step_df = step_df[step_params_names + ['om','Rm']].rename(columns={'om':'step_om','Rm':'step_Rm'})
        param_df = param_df[pair_params_names + ['om','Rm']].rename(columns={'om':'pair_om','Rm':'pair_Rm'})
        pairs_df = pairs_df[['resid1','segid1','restype1','origin1','R_frame1','resid2','segid2','restype2','origin2','R_frame2']]
        return pd.concat([pairs_df, param_df,step_df],axis=1)
    param_df = pd.concat([pairs_df[['resid1','segid1','restype1','resid2','segid2','restype2']],
                                   param_df[pair_params_names],step_df[step_params_names]],axis=1)
    return param_df