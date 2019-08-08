#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
try:
    from VMD import *
    from Molecule import *
    from atomsel import *
    from animate import *
    NOVMD=False
except:
    print "No VMD support found"
    NOVMD=True
    
from force_constants import get_consts
from scipy.optimize import minimize, basinhopping
import uuid
from prody import *
confProDy(verbosity='none')

import cStringIO

import numpy as np

__author__="Armeev Grigoriy,Alexey Shaytan"

class py3dna():
    def __init__(self,pdbfilename=None,VMD_atomsel=None,tempdir=None,path='/home/armeev/Software/Source/x3dna-v2.1'):
        if (tempdir==None):
            os.mkdir('temp')
            self.TEMP='temp'
        else:
            self.TEMP=tempdir
        
        self.set_3dna_path(path)
                
        self.AVERAGE,self.FORCE_CONST,self.DISP,self.lbl1,self.lbl2=get_consts()
        self.lbl1_ref=self.lbl1.select(''' name C5' C4' C3' C2' C1' O4' ''')
        self.lbl2_ref=self.lbl2.select(''' name C5' C4' C3' C2' C1' O4' ''')
        self.lbl1_com=self.lbl1.select(''' name COM ''')
        self.lbl2_com=self.lbl2.select(''' name COM ''')
        #confProDy(verbosity='none')
        
        #copying some files for backbone reconstruction
        cmd=self.P_X3DNA_utils + ' cp_std BDNA'   
        p = subprocess.Popen(cmd,shell=True,cwd=self.TEMP,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        self.pairs_id=''
        self.distCoef=0.1
        
        
        if VMD_atomsel!=None:
            if NOVMD:
                print "VMD Libs aren't avaiable"
                return
            DNA=VMD_atomsel
            pairparam=self.X3DNA_find_pair(DNA_atomsel=DNA)
            
            
            
        elif pdbfilename!=None:            
            pairparam=self.X3DNA_find_pair(pdbfile=pdbfilename)
        else:
            print "You must provide VMD atomsel or pdb filename"
            return
        
        self.par_header,self.par_pairs,self.par_frame=self.X3DNA_analyze(pairparam)
        if VMD_atomsel!=None:
            if NOVMD:
                print "VMD Libs aren't avaiable"
                return
            self.frame_to_pdb(self.par_frame,'init_rebuilt.pdb')
            self.mol=Molecule()                    
            self.mol.load(self.TEMP+'/init_rebuilt.pdb')
            
        self.num_of_res=self.par_frame.shape[0]
        self.set_movable_bp([[1,self.num_of_res]])

    def set_3dna_path(self,path='/home/armeev/Software/Source/x3dna-v2.1'):
        self.P_X3DNA_DIR=path
        self.P_X3DNA_analyze=self.P_X3DNA_DIR+'/bin/analyze'
        self.P_X3DNA_find_pair=self.P_X3DNA_DIR+'/bin/find_pair'
        self.P_X3DNA_utils=self.P_X3DNA_DIR+'/bin/x3dna_utils'
        self.P_X3DNA_rebuild=self.P_X3DNA_DIR+'/bin/rebuild'
        os.environ['X3DNA']=self.P_X3DNA_DIR
        
    def set_pairs_list_and_dist(self,pairs_list,dist_list):
        '''
        This function sets constraints for distances between b.p.
        provide pairs and length list like
        [[1,147],[3,18],[6,22]],[20,30,40]
        for 3 distances and their lengths 
        WARNING - one bp can't be in two distance pairs
        '''
        pairs_id=[]
        for pair in pairs_list:                    
            bp1=[pair[0],self.num_of_res*2-pair[0]+1]
            bp2=[pair[1],self.num_of_res*2-pair[1]+1]
            pairs_id.append([bp1,bp2])
        self.pairs_id=np.array(pairs_id)
        self.pairs_dist=dist_list
        if not(NOVMD):
            self.vmd_pairs_sel=[]
            for pair in self.pairs_id:
                bp1=atomsel('resid ' + str(pair[0][0]) + ' ' + str(pair[0][1]))
                bp2=atomsel('resid ' + str(pair[1][0]) + ' ' + str(pair[1][1]))
                self.vmd_pairs_sel.append([bp1,bp2])
    
    def set_movable_bp(self,listofpairs):
        '''
        This function sets b.p. as movable e.g. only them will
        be affected during minimization
        If you want bp number 1 to 8, 15 to 18, 118 to 147 (inclusive)
        to be movable provide list like
        [[1,8],[15,18],[118,147]]
        other pairs's variables will not change during minimisation
        NOTE! - numeration starts from 1 ALL the time.
        if you aren't shure about numbers create PDB with frame_to_pdb()
        and check them        
        '''
        movable_bp=[]
        for bp in listofpairs:
            movable_bp.append(range(bp[0],bp[1]))
        self.movable_bp=np.hstack(movable_bp)
    
    def X3DNA_find_pair(self,DNA_atomsel=None,pdbfile=None):
        """Performs the analysis using X3DNA

        Parameters
        ----------
        DNA_atomsel - DNA segments selected by atomsel command in VMD.
        (NOT AtomSel!)
        
        Return
        --------
        string from sdoutput of find_pair
        """
        if (DNA_atomsel != None):
            #At first we need to makup a couple of unique file names
            unique=str(uuid.uuid4())
            pdb = unique+'.pdb'
            outf = unique

            print("Writing coords to "+pdb)
            DNA_atomsel.write('pdb',self.TEMP+'/'+pdb)
        if pdbfile != None:
            pdb = os.path.abspath(pdbfile)
            
        cmd=self.P_X3DNA_find_pair+' '+pdb+' stdout'
        #print(cmd)
        p = subprocess.Popen(cmd,shell=True,cwd=self.TEMP,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # wait for the process to terminate
        stdout, err = p.communicate()
        errcode = p.returncode
        #print('OUT:'+out)
        print('ERR:'+err)
        return(stdout)

    def X3DNA_analyze(self,ref_fp_id):
        """Performs the analysis using X3DNA

        Parameters
        ----------
        DNA_atomsel - DNA segments selected by atomsel command in VMD.
        (NOT AtomSel!)
        ref_fp_id - this is output id from X3DNA_find_pair function,
        
        Return
        --------
        header - list with header of .par file from analyse
        pairtypes - list with b.p
        par_frame - numpy array with pb params bp in rows params in columns
        """

        #Now we can run X3DNA_analyze
        cmd=self.P_X3DNA_analyze + ' stdin'
        p = subprocess.Popen(cmd,shell=True,cwd=self.TEMP,stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=ref_fp_id)
        print('OUT:'+out+err)

        header,pairtypes,par_frame=self.par_to_frame(self.TEMP+'/'+'bp_step.par')

        return header,pairtypes,par_frame

    def par_to_frame(self,file):
        """
        Parse bp_step.par file as output by X3DNA
        Get a data frame with base and base-pair step
        parameters

        Note that in each line of the data frame there are parameters
        for some base pair and base pair step that preceeds (!) this base-pair.
        I.e. in the first line no base pair step parameters are specified!

        offest - offset for DNA numbering.
        """
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
        
    def frame_to_par(self,frame,filename):
        """
        this fuction writes analog of bp_step.par file from frame to filename.
    
        """
        string=''
        for line in self.par_header:
            for word in line[:-1]:
                string+=(str(word) + '\t')
            string+='\n'

        data1=np.append(self.par_pairs,frame.transpose().astype(str)).reshape(frame.shape[1]+1,-1).transpose()
       
        np.savetxt(filename,data1,fmt='%6s',header=string[:-1],comments='  ',delimiter='\t')

    def frame_to_pdb(self,frame,filename):
        """
        this fuction Builds pdb from frame to filename
        """
        #creating par file
        self.frame_to_par(frame,self.TEMP+'/tempfile.par')
        #rebuilding
        cmd=self.P_X3DNA_rebuild+' -atomic tempfile.par ' + filename
        p = subprocess.Popen(cmd,shell=True,cwd=self.TEMP,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        print out,err

    def frame_to_vmd(self,frame,dist=False,remove=True):
        """
        this fuction Builds pdb from frame to filename
        """
        self.frame_to_pdb(frame,'temp_minim.pdb')
        if remove:
            tempmol=Molecule()                        
            tempmol.load(self.TEMP+'/temp_minim.pdb')
            result=np.zeros(len(self.pairs_id))
            i=0
            for pair in self.pairs_id:
                #This should work for top mol
                bp1=atomsel('resid ' + str(pair[0][0]) + ' ' + str(pair[0][1]))
                bp2=atomsel('resid ' + str(pair[1][0]) + ' ' + str(pair[1][1]))
                result[i]=np.linalg.norm(np.array(bp1.center())-np.array(bp2.center()))
                i+=1
            evaltcl('display update')
            tempmol.delete()
            return result
        else:
            self.mol.load(self.TEMP+'/temp_minim.pdb')
            if dist:
                result=np.zeros(len(self.vmd_pairs_sel))
                i=0
                for pairsel in self.vmd_pairs_sel:
                    result[i]=np.linalg.norm(np.array(pairsel[0].center())-np.array(pairsel[1].center()))
                    
                    i+=1
                evaltcl('display update')
                
                return result
            evaltcl('display update')
        
    def frame_to_dist(self,frame,use_dyes=False):
        '''
        returns np.array of distances between pairs set by self.set_pairs_list()
        '''
        #creating par file
        selection=' and ((resname DC DT and name N1) or (resname DG DA and name N9))'
        self.frame_to_par(frame,self.TEMP+'/tempfile.par')
        cmd=self.P_X3DNA_rebuild+' -atomic tempfile.par stdout'
        p = subprocess.Popen(cmd,shell=True,cwd=self.TEMP,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pdb_str, err = p.communicate()
        #self.frame_to_pdb(frame,'temp_minim.pdb')
        tempAtomGrp=parsePDBStream(cStringIO.StringIO(pdb_str))
        #tempAtomGrp=parsePDB(self.TEMP+'/temp_minim.pdb')
        result=np.zeros(len(self.pairs_id))
        if not(use_dyes):
            i=0
            for pair in self.pairs_id:
                c1=tempAtomGrp.select('resid ' + str(pair[0][0]) + ' ' + str(pair[0][1])
                     + selection).getCoords().sum(axis=0)/2.0
                c2=tempAtomGrp.select('resid ' + str(pair[1][0]) + ' ' + str(pair[1][1])
                     + selection).getCoords().sum(axis=0)/2.0
                result[i]=np.linalg.norm(c1-c2)
                i+=1
        else:
            i=0
            for pair in self.pairs_id:
                '''resid 1 and name C5' C4' C3' C2' C1' O4' '''
                target1=tempAtomGrp.select('resid ' + str(pair[0][0]) +  ''' and name C5' C4' C3' C2' C1' O4' ''')
                target2=tempAtomGrp.select('resid ' + str(pair[1][1]) +  ''' and name C5' C4' C3' C2' C1' O4' ''')
                calcTransformation(self.lbl1_ref,target1).apply(self.lbl1)
                calcTransformation(self.lbl2_ref,target2).apply(self.lbl2)
                
                result[i]=np.linalg.norm(self.lbl1_com.getCoords()-self.lbl2_com.getCoords())
                i+=1
        
        #print result       
        return result
        
    def frame_to_energy(self,frame,usepairs=False):
        '''
        returns energy of dna conformation, calculated with force params
        usepairs flag enables calculation of distances between pairs set
        by self.set_pairs_list()
        '''
        self.copy_frame=self.par_frame.copy()
        energy=self.calc_energy(frame[self.movable_bp,6:12],usepairs=usepairs)
        return energy
        
    def run_minimize(self,frame=None,usepairs=False,use_dyes=False,vmdload=False,save_deriv=False,
        options={'maxiter':20,'maxfev': 20,'xtol' :2.0, 'disp': True},method='Powell',jac=False):
        '''
        Preformes minimization of energy
        NOTE! it only works with b.p. steps selected with set_movable_bp()
        usepairs flag enables calculation of distances between pairs set
        by self.set_pairs_list()

        '''
        if type(frame)=='NoneType':
            frame=self.par_frame

        self.copy_frame=self.par_frame.copy()
        x0=frame[self.movable_bp,6:12].flatten().astype(float)
        if NOVMD:
            #res = minimize(self.calc_energy, x0, (usepairs,False,use_dyes,save_deriv), method=method,options=options,jac=jac)
            res = basinhopping(self.calc_energy, x0, niter=options['niter'],T=options['T'],minimizer_kwargs={'args':(usepairs,False,use_dyes,save_deriv), 'method':method})
        else:
            res = minimize(self.calc_energy, x0, (usepairs,vmdload,use_dyes,save_deriv), method=method,options=options,callback=self.subframe_to_vmd)
        local_frame=self.par_frame.copy()
            
        local_frame[self.movable_bp,6:12]=np.array(res.x).reshape(-1,6)
        return local_frame,res
    
    def subframe_to_vmd(self,subframe):
        local_frame=self.copy_frame
        local_frame[self.movable_bp,6:12]=np.array(subframe).reshape(-1,6)
        self.frame_to_vmd(local_frame,remove=False)
    
    def subframe_to_PDB(self,subframe):
        local_frame=self.copy_frame
        local_frame[self.movable_bp,6:12]=np.array(subframe).reshape(-1,6)
        self.frame_to_pdb(local_frame,'temp_minim.pdb')
        
    def calc_energy(self,frame,usepairs=False,vmdload=False,use_dyes=False,save_deriv=False):
        '''
        !!!DO NOT USE THIS FUNCTION DIRECTLY!!!
        instead use frame_to_energy()
        returns energy of dna conformation, calculated with force params
        usepairs flag enables calculation of distances between pairs set
        by self.set_pairs_list()
        '''
        local_frame=self.copy_frame
        local_frame[self.movable_bp,6:12]=np.array(frame).reshape(-1,6)
        try:
            subframe=local_frame[1:,6:12].astype(float)
        except:
            local_frame[1:,6:12].tolist()
            return 
        dif_matrix=[]
        force_matrix=[]
        for i in range(len(self.par_pairs)-1):
            step=str(self.par_pairs[i][0]+self.par_pairs[i+1][0])
            a=subframe[i].astype(float)-self.AVERAGE[step].astype(float)
            dif_matrix.append(np.tile(a,(6,1)).transpose()*np.tile(a,(6,1)))
            force_matrix.append(self.FORCE_CONST[step])
        
        if not(usepairs):
            result = float(np.multiply(dif_matrix, force_matrix).sum()/2.0)
            if vmdload:
                self.frame_to_vmd(local_frame)
        else:
            if not(vmdload):
                dists=self.frame_to_dist(local_frame,use_dyes=use_dyes)
                result = (float(np.multiply(dif_matrix, force_matrix).sum()/2.0)            
                 + self.distCoef*np.power(self.pairs_dist-dists,2).sum())
            else:
                result = (float(np.multiply(dif_matrix, force_matrix).sum()/2.0)
                 + self.distCoef*np.power(self.pairs_dist-self.frame_to_vmd(local_frame,dist=True,remove=not(save_deriv)),2).sum())
        del local_frame
        if not(usepairs):        
            print '\rEnergy: ' + str(result),
        else:
            print '\rEnergy: ' + str(result) + ' Dist: ' + str(dists),
        return result
