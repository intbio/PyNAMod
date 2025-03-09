import torch

class Geometry_Functions:
    '''This class contains functions to rebuild reference frames and ori from local DNA parameters for full strucuture or partially with rotation and to rebuild local parameters from reference frames amd origins. This class is supposed to be used as a super class for Geometrical_Parameters.'''
    def rebuild_ref_frames_and_ori(self,start_index = 0, stop_index = None, start_ref_frame = None,start_origin = None,rebuild_proteins=False):
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
            self.origins[start_index] = start_origin = torch.zeros(3)
        else:
            self.origins[start_index] =  start_origin
        
        for i,r in enumerate(r2):
            self.ref_frames[start_index+i+1] = torch.mm(self.ref_frames[start_index+i],r)
        self.origins[start_index+1:stop_index] = torch.cumsum(torch.matmul(o2,torch.transpose(self.ref_frames[start_index:stop_index-1],2,1)),dim=0).reshape(-1,1,3)+start_origin
        
        
    
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
        prev_o = self.origins[change_index].clone()
        self.rebuild_ref_frames_and_ori(change_index-1,change_index+1,self.ref_frames[change_index-1],self.origins[change_index-1],rebuild_proteins=False)
        if change_index != self.len:
            rot_matrix = self.ref_frames[change_index].mm(prev_R.T)
            self.__rotate_R(change_index,rot_matrix)
            self.__transform_ori(change_index,rot_matrix,prev_o,self.origins[change_index])

    def __rotate_R(self,change_index,rot_matrix):
        self.ref_frames[change_index:] = rot_matrix.reshape(1,3,3).matmul(self.ref_frames[change_index:])
        R_frames = self.ref_frames[change_index:]
        #if torch.linalg.norm(R_frames,axis=1).mean() != 1:
        #    self.ref_frames[change_index:] = abs(torch.linalg.qr(R_frames).Q)*R_frames.sign()

    def __transform_ori(self,change_index,rot_matrix,prev_ori,changed_ori):
        stop = self.origins.shape[0]
        if hasattr(self,'prot_ind'):
            for ref_ind in sorted(self.prot_ind)[::-1]:
                if ref_ind < change_index:
                    stop = self.prot_ind[ref_ind][0]
                    break
        self.origins[change_index+1:stop] -= prev_ori
        self.origins[change_index+1:stop] = self.origins[change_index+1:stop].matmul(rot_matrix.T) + changed_ori

    
    
    
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