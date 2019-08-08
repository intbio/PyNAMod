from pynamod.energy_funcs import *
from pynamod.energy_constants import get_consts_olson_98
from scipy.optimize import basinhopping

AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()

def run_basinhopping(full_par_frame,pairs,movable_steps,FORCE_CONST,AVERAGE,
                     save_hops=False,disp=True,options={'niter':1,'T': 1000},method='Powell',jac=False):
    '''
    Preformes minimization of energy with the scipy basinhopping algorythm
    input:
    full_par_frame - numpy array with N x 12 b.p. par frame
    pairs - list of pairtypes e.g.  ['A-T', 'T-A', 'C-G',...,'A-T']
    movable_steps - list with b.p. step indexes which should be changed during optimization
                    can be obtained with utils.get_movable_steps()
    FORCE_CONST   - dict with b.p. bending force constants (see energy constants.get_consts_olson_98)
    AVERAGE       - dict with average b.p. step values (see energy constants.get_consts_olson_98)
    
    !!!!!TODO - continue documentation
    
    returns:
    result_frame  - numpy array with N x 12 b.p. par frame after optimization
    res           - scipy optimizer result object
    '''
    from pynamod.energy_funcs import _calc_bend_energy
    force_matrix=get_force_matrix(pairs,movable_steps,FORCE_CONST)
    average_bpstep_frame=get_force_matrix(pairs,movable_steps,AVERAGE)
    bpstep_frame=get_bpstep_frame(full_par_frame,movable_steps)
    x0=bpstep_frame.flatten()
    res = basinhopping(_calc_bend_energy, x0, niter=options['niter'],T=options['T'],disp=disp,
                       minimizer_kwargs={'args':(force_matrix,average_bpstep_frame), 'method':method})
   
    result_frame=full_par_frame.copy()

    result_frame[movable_steps+1,6:12]=np.array(res.x).reshape(-1,6)
    return result_frame,res
