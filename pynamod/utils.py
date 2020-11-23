import numpy as np
def get_movable_steps(movable_bp):
    '''
        This function sets b.p. as movable e.g. only them will
        be affected during minimization
        If you want bp number 1 to 8, 15 to 18, 118 to 147 (inclusive)
        to be movable provide list like
        [[1,8],[15,18],[118,147]]
        other pairs's variables will not change during minimisation
        NOTE! - numeration starts from bp 1 ALL the time.
    '''
    movable_bpstep=[]
    for bp in movable_bp:
        movable_bpstep.append(range(*bp))
    return(np.hstack(movable_bpstep)-1)

W_C_pair={'A':'T','G':'C','C':'G','T':'A'}