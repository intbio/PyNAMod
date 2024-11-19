import torch


class Geometrical_Parameters:
    def __init__(self,local_params = None, ref_frames = None, origins = None,pair_params=False,auto_rebuild=True):

        self.pair_params = pair_params
                
        init_from_local_params = local_params is not None
        init_from_r_and_o = ref_frames is not None and origins is not None
        
        self._auto_rebuild_sw = False
        self.auto_rebuild = auto_rebuild
        
                
        if init_from_r_and_o and init_from_local_params:     
            if not origins.dtype == ref_frames.dtype == local_params.dtype:
                raise TypeError("Dtypes don't match")
            self.ref_frames = ref_frames
            self.origins = origins
            self.local_params = local_params    
            self.dtype = origins.dtype
        elif init_from_r_and_o:
            if origins.dtype != ref_frames.dtype:
                raise TypeError("Origins and reference frames dtypes don't match")
            
            self.ref_frames = ref_frames
            self.origins = origins

            self.dtype = origins.dtype
            self.len = origins.shape[0]
            self.local_params = torch.zeros(self.len,6,dtype=self.dtype)
            self.rebuild('rebuild_local_params')
            
        elif init_from_local_params:
            self.local_params = local_params
            self.dtype = local_params.dtype
            self.len = local_params.shape[0]
            self.ref_frames = torch.zeros((self.len,3,3),dtype=self.dtype)
            self.origins = torch.zeros((self.len,3),dtype=self.dtype)
            self.rebuild('rebuild_ref_frames_and_ori')
        
        else:
            raise TypeError('Geometrical_parameters should be initialized with local parameters or reference frames and origins')
       
        self._auto_rebuild_sw = auto_rebuild

    def copy(self):
        return Geometrical_Parameters(local_params = self.local_params.clone(), ref_frames = self.ref_frames.clone(),
                                      origins = self.origins.clone(),pair_params=self.pair_params)
    
    
    def rebuild(self,rebuild_func_name,*args,**kwards):
        self._auto_rebuild_sw = False

        getattr(self,rebuild_func_name)(*args,**kwards)
        
        self._auto_rebuild_sw = self.auto_rebuild
    
    def rebuild_ref_frames_and_ori(self,start_index = 0, stop_index = None, start_ref_frame = None,start_origin = None):
        
        if stop_index is None:
            stop_index = self.len

        
        dist_params = self.local_params[start_index + 1:stop_index,:3]
        angle_params = torch.deg2rad(self.local_params[start_index + 1:stop_index,3:])


        gamma = torch.norm(angle_params[:,:2],dim=1)
        cos_phi =  angle_params[:,1]/gamma
        phi = torch.arccos(cos_phi)
        phi[angle_params[:,0]<0] *= -1

        rm = self.__get_r_mat(angle_params[:,2]/2 - phi,gamma/2,phi)

        r2 = self.__get_r_mat(angle_params[:,2]/2 - phi,gamma,angle_params[:,2]/2 + phi)
        o2 = torch.matmul(dist_params.reshape(-1,1,3),torch.transpose(rm,2,1))
        
        if start_ref_frame is None:
            self.ref_frames[start_index] = torch.eye(3)
        else:
            self.ref_frames[start_index] = start_ref_frame
            
        if start_origin is None:
            self.origins[start_index] = torch.zeros(3)
        else:
            self.origins[start_index] =  start_origin
        
        for i,r in enumerate(r2):
            self.ref_frames[start_index+i+1] = torch.mm(self.ref_frames[start_index+i],r)


        self.origins[start_index+1:stop_index] = torch.cumsum(torch.matmul(o2,torch.transpose(self.ref_frames[start_index:stop_index-1],2,1)),axis=0)
        
        
    
    def rebuild_local_params(self,start_index=0):
        
        R1,R2 = self.ref_frames[start_index:-1],self.ref_frames[start_index + 1:]
        z1 = R1[:,:,2]
        z2 = R2[:,:,2]
        if self.pair_params:
            sl = (z1 * z2).sum(dim = 1) < 0
            R1 = R1.clone()
            R1[sl,:,1:] *= -1
                
        o1,o2 = self.origins[start_index:-1],self.origins[start_index + 1:]

        hinge = torch.linalg.cross(z1,z2)

        hinge /= torch.norm(hinge,dim=1).reshape(-1,1)
        RollTilt = torch.arccos((z1*z2).sum(dim = 1))
        R_hinge = self.__rmat(hinge,-0.5*RollTilt)
        R2p = torch.matmul(R_hinge,R2)

        R1p = torch.matmul(self.__rmat(hinge,0.5*RollTilt),R1)

        self.Rm = (R1p+R2p)/2.0
        vectors_norm = torch.norm(self.Rm,dim=2).reshape(-1,1,3)
        self.Rm /= vectors_norm

        self.om=(o1+o2)/2.0
        self.local_params[start_index + 1:,:3] = torch.matmul((o2-o1).reshape(-1,1,3),self.Rm).reshape(-1,3)

        twist = torch.arccos((R1p[:,:,0] * R2p[:,:,0]).sum(dim = 1))
        twist[twist.isnan()] = 0
        twist_sign = (torch.linalg.cross(R1p[:,:,1],R2p[:,:,1])*self.Rm[:,:,2]).sum(dim = 1)
        twist[twist_sign < 0] *= -1
        self.local_params[start_index + 1:,5] = torch.rad2deg(twist)

        phi_cos = (hinge * self.Rm[:,:,1]).sum(dim = 1)
        phi = torch.arccos(phi_cos)
        phi[phi.isnan()] = 0
        phi_sign = (torch.linalg.cross(hinge,self.Rm[:,:,1])*self.Rm[:,:,2]).sum(dim = 1)
        phi[phi_sign < 0] *= -1

        self.local_params[start_index + 1:,3] = torch.rad2deg(RollTilt*phi.sin())
        self.local_params[start_index + 1:,4] = torch.rad2deg(RollTilt*phi.cos())
        
    
    def rotate_ref_frames_and_ori(self,change_index):
        prev_R = self.ref_frames[change_index].clone()
        self.rebuild_ref_frames_and_ori(change_index-1,change_index+1,self.ref_frames[change_index-1],self.origins[change_index-1])
        changed_R,changed_ori = self.ref_frames[change_index],self.origins[change_index]

        rot_matrix = changed_R.mm(prev_R.T)

        self.ref_frames[change_index:] = self.__rotate_R(self.ref_frames[change_index:],rot_matrix)
        self.origins[change_index+1:] = self.__transform_ori(self.origins[change_index+1:],rot_matrix,
                                                        self.origins[change_index],changed_ori)
        self.origins[change_index] = changed_ori


    def __rotate_R(self,R_frames,rot_matrix):
        R_frames = rot_matrix.reshape(1,3,3).matmul(R_frames)
        if torch.linalg.norm(R_frames,axis=1).mean() != 1:
            R_frames = abs(torch.linalg.qr(R_frames).Q)*R_frames.sign()

        return R_frames

    def __transform_ori(self,ori,rot_matrix,old_ori,changed_ori):
        ori -= old_ori
        ori = ori.matmul(rot_matrix.T) + changed_ori
    
        return ori
    
    
    
    def __get_trig(self,angle):
        return angle.cos(),angle.sin()
    
    
    
    def __get_r_mat(self,angle1,angle2,angle3):
        cos1,sin1 = self.__get_trig(angle1)
        cos2,sin2 = self.__get_trig(angle2)
        cos3,sin3 = self.__get_trig(angle3)

        a = sin1*sin3
        b = cos1*cos3
        c = sin1*cos3
        d = cos1*sin3
        chunk1 = torch.stack([b,-d,
                           c,-a
                          ],dim=1)
        chunk1 = chunk1*cos2.reshape(-1,1) + torch.stack([-a,-c,d,b],dim=1)

        chunk2 = torch.stack([cos1,sin1,-cos3,sin3],dim=1) * sin2.reshape(-1,1)
        r_mat = torch.cat([chunk1,chunk2,cos2.reshape(-1,1)],dim=1)[:,[0,1,4,2,3,5,6,7,8]]

        return r_mat.reshape(-1,3,3)

        
    def __rmat(self,axis, phi):
        u1 = axis[:,0].reshape(-1,1)
        u2 = axis[:,1].reshape(-1,1)
        u3 = axis[:,2].reshape(-1,1)
        c = phi.cos().reshape(-1,1)
        s = phi.sin().reshape(-1,1)
        return torch.cat([c+(1-c)*u1*u1, (1-c)*u1*u2-u3*s, (1-c)*u1*u3+u2*s,
                         (1-c)*u1*u2+u3*s,  c+(1-c)*u2*u2,(1-c)*u2*u3-u1*s,
                         (1-c)*u1*u3-u2*s, (1-c)*u2*u3+u1*s,   c+(1-c)*u3*u3],dim=1).reshape(-1,3,3)
    
    
    def __getitem__(self,sl):
        if isinstance(sl,slice):
            if sl.step is not None:
                if sl.step < 0:
                    self._auto_rebuild_sw = False
                    self.origins = self.origins[::-1]
                    self._auto_rebuild_sw = self.auto_rebuild
                    self.ref_frames = self.ref_frames[::-1]
                    sl = slice(sl.start,sl.stop,-1*sl.step)
        return Geometrical_Parameters(local_params = self.local_params[sl], ref_frames = self.ref_frames[sl],
                                      origins = self.origins[sl],pair_params=self.pair_params)
    
    
    def __getter(self,attr):
        return getattr(self,attr)
    
    def __setter(self,value,attr,rebuild_func_name):
        setattr(self,attr,mod_Tensor(value,self))
        if self._auto_rebuild_sw:
            self.rebuild(rebuild_func_name)

        
    local_params = property(fget=lambda self: self.__getter(attr = '_local_params'),
                            fset=lambda self,value: self.__setter(value,attr = '_local_params',rebuild_func_name='rebuild_ref_frames_and_ori'))
    ref_frames = property(fget=lambda self: self.__getter(attr = '_ref_frames'),
                          fset=lambda self,value: self.__setter(value,attr = '_ref_frames',rebuild_func_name='rebuild_local_params'))
    origins = property(fget=lambda self: self.__getter(attr = '_origins'),
                       fset=lambda self,value: self.__setter(value,attr = '_origins',rebuild_func_name='rebuild_local_params'))
    

class mod_Tensor(torch.Tensor):
    geom_class = None
    def __new__(cls, x,geom_class, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self,x,geom_class,*args,**kwards):
        self.geom_class = geom_class
        shape = self.shape
        if shape[1] == 6:
            self.type = 'local_params'

        elif shape[1] == 3:
            self.type = 'r_or_ori'



    def __getitem__(self, sl):
        it = super().__getitem__(sl)
        it.geom_class = self.geom_class
        it.type = self.type
        return it

    def __setitem__(self, sl, value):

        super().__setitem__(sl,value)
        
        if self.geom_class is not None:
            if self.geom_class._auto_rebuild_sw:
        
                if isinstance(sl,tuple):
                    sl = sl[0]

                if isinstance(sl,slice):
                    if self.type == 'r_or_ori':
                        self.geom_class.rebuild('rebuild_local_params')
                    elif self.type == 'local_params':
                        self.geom_class.rebuild('rebuild_ref_frames_and_ori')

                elif isinstance(sl,int):
                    if self.type == 'r_or_ori':
                        self.geom_class.rebuild('rebuild_local_params',start_index=sl)
                    elif self.type == 'local_params':
                        self.geom_class.rebuild('rotate_ref_frames_and_ori',sl)
                    