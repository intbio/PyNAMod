import numpy as np
from  numpy.lib.recfunctions import append_fields

def parse_bp_par_file(file):
    """
    Parse bp_step.par file as output by X3DNA
    Get a data frame with base and base-pair step
    parameters

    Note that in each line of the data frame there are parameters
    for some base pair and base pair step that preceeds (!) this base-pair.
    I.e. in the first line no base pair step parameters are specified!
    """
    print("Processing ", file)
    params=[]
    with open(file,'r') as f:
        for line in f:
            params.append(line.split()) 
    header=params[0:3]
    pairtypes=tuple(zip(*params[3:]))[0]
    par_frame=np.array(tuple(zip(*params[3:]))[1:]).transpose().astype(float)
    return(header,pairtypes,par_frame)

def write_bp_par_file(pairtypes,frame,filename):
    
    """
    this fuction writes analog of bp_step.par file from frame to filename.

    """
    header=f'''{len(pairtypes):4d} # base-pairs
   0 # ***local base-pair & step parameters***
#        Shear    Stretch   Stagger   Buckle   Prop-Tw   Opening     Shift     Slide     Rise      Tilt      Roll      Twist'''

    a = np.rec.array(np.array(pairtypes), dtype=[('pair', '<U3')])
    variables=['Shear', 'Stretch','Stagger','Buckle','Prop-Tw','Opening',  'Shift',  'Slide', 'Rise','Tilt','Roll','Twist']
    for i,var in enumerate(variables):
        a = append_fields(a, var, frame[:,i], usemask=False, dtypes=[np.float32])
    fmt=['%-4s']+['%10.3f']*12
    np.savetxt(filename,a,fmt=fmt,header=header,comments='',delimiter='')

def ref_frames_to_array(fname):
    '''
    This funcrion parses 3dna ref_frame.dat file (from analyze command)
    and returns the locations of base pair reference frames with pair types:
    
    ref_frames - Nx4x4 numpy array
    x_axis y_axis z_axis
    ----------------------
    x_a   |Y_a   |Z_a   |0
    x_b   |Y_b   |Z_b   |0
    x_c   |Y_c   |Z_c   |0
    orig_x|orig_y|orig_z|0
    
    pairs - list with pair types
    e.g. ['A-T', 'T-A', 'C-G',...,'A-T']
    '''
    with open(fname) as df:
        num_pairs=int(df.readline().split()[0])
        ref_frames=np.zeros([num_pairs,4,4])
        pairtypes=[]
        for n, line in enumerate(df):
            if (n)%5==0:
                pairtypes.append(line.split()[2])
            elif (n-1)%5==0:
                ref_frames[(n-1)//5,3,:3]=line.split()[0:3]
            elif (n-2)%5==0:
                ref_frames[(n-2)//5,:3,0]=line.split()[0:3]
            elif (n-3)%5==0:
                ref_frames[(n-3)//5,:3,1]=line.split()[0:3]
            elif (n-4)%5==0:
                ref_frames[(n-4)//5,:3,2]=line.split()[0:3]
    return ref_frames,pairtypes