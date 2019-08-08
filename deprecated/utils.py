import numpy as np

def par_to_frame(file):
    print "Processing ", file
    params=[]
    print file
    with open(file,'r') as f:
        for line in f:
            params.append(line.split())    
    header=params[0:3]
    pairtypes=zip(*params[3:])[0]
    par_frame=np.array(zip(*params[3:])[1:]).transpose()
    return header,pairtypes,par_frame

def frame_to_par(filename,header,par_pairs,frame):
    """
    this fuction writes analog of bp_step.par file from frame to filename.

    """
    string=''
    for line in header:
        for word in line[:-1]:
            string+=(str(word) + '\t')
        string+='\n'

    data1=np.append(par_pairs,frame.transpose().astype(str)).reshape(frame.shape[1]+1,-1).transpose()

    np.savetxt(filename,data1,fmt='%6s',header=string[:-1],comments='  ',delimiter='\t')

complement={'A':'T', 'T':'A', 'C':'G', 'G':'C'}