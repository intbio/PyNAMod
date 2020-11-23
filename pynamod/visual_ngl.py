import numpy as np
import nglview as nv
def show_ref_frames(bp_frames,view=None,spheres=True,arrows=False,diamonds=False,boxes=True,bp_colors_mask=None):
    if view is None:
        view=nv.NGLWidget()
    dx=2
    dy=4
    dz=1
    if spheres:
        if bp_colors_mask is None:
            view.shape.add_buffer('sphere',position=bp_frames[:,3,:3].flatten().tolist(),color=[1,1,0]*bp_frames.shape[0],radius=[3]*bp_frames.shape[0])
        else:
            bp_colors=np.ones((bp_frames.shape[0],3))
            bp_colors[bp_colors_mask,1]=0
            bp_colors[bp_colors_mask!=1,2]=0
            view.shape.add_buffer('sphere',position =bp_frames[:,3,:3].flatten().tolist(),color=bp_colors.flatten().tolist(),radius=[3]*bp_frames.shape[0])
        view.shape.add_buffer('cylinder',
                              position1=bp_frames[:,3,:3][:-1].flatten().tolist(),
                              position2=bp_frames[:,3,:3][1:].flatten().tolist(),
                              color=[1,1,0]*(bp_frames.shape[0]-1),radius=[0.5]*(bp_frames.shape[0]-1))
    if arrows:
        view.shape.add_buffer('arrow',
                              position1=bp_frames[:,3,:3].flatten().tolist(),
                              position2=(bp_frames[:,3,:3]+3*bp_frames[:,:3,0]).flatten().tolist(),
                              color=[1,0,0]*bp_frames.shape[0],radius=[0.3]*bp_frames.shape[0])
        view.shape.add_buffer('arrow',
                              position1=bp_frames[:,3,:3].flatten().tolist(),
                              position2=(bp_frames[:,3,:3]+3*bp_frames[:,:3,1]).flatten().tolist(),
                              color=[0,1,0]*bp_frames.shape[0],radius=[0.3]*bp_frames.shape[0])
        view.shape.add_buffer('arrow',
                              position1=bp_frames[:,3,:3].flatten().tolist(),
                              position2=(bp_frames[:,3,:3]+3*bp_frames[:,:3,2]).flatten().tolist(),
                              color=[0,0,1]*bp_frames.shape[0],radius=[0.3]*bp_frames.shape[0])
    if diamonds:
        z0=bp_frames[:,3,:3]-dz*bp_frames[:,:3,2]
        z1=bp_frames[:,3,:3]+dz*bp_frames[:,:3,2]
        y0=bp_frames[:,3,:3]-dy*bp_frames[:,:3,1]
        y1=bp_frames[:,3,:3]+dy*bp_frames[:,:3,1]
        x0=bp_frames[:,3,:3]-dx*bp_frames[:,:3,0]
        x1=bp_frames[:,3,:3]+dx*bp_frames[:,:3,0]
        coords=np.hstack((x0,y0,z0,x0,y0,z1,
                   x0,y1,z0,x0,y1,z1,
                   x1,y0,z0,x1,y0,z1,
                   x1,y1,z0,x1,y1,z1,)).flatten()
        color=[1,1,0]*(coords.size//3)
        view.shape.add_mesh(coords.tolist(), color, )
    if boxes:
        p1=bp_frames[:,3,:3]-dx*bp_frames[:,:3,0]-dy*bp_frames[:,:3,1]-dz*bp_frames[:,:3,2]
        p2=bp_frames[:,3,:3]-dx*bp_frames[:,:3,0]-dy*bp_frames[:,:3,1]+dz*bp_frames[:,:3,2]
        p3=bp_frames[:,3,:3]-dx*bp_frames[:,:3,0]+dy*bp_frames[:,:3,1]-dz*bp_frames[:,:3,2]
        p4=bp_frames[:,3,:3]-dx*bp_frames[:,:3,0]+dy*bp_frames[:,:3,1]+dz*bp_frames[:,:3,2]
        p5=bp_frames[:,3,:3]+dx*bp_frames[:,:3,0]-dy*bp_frames[:,:3,1]-dz*bp_frames[:,:3,2]
        p6=bp_frames[:,3,:3]+dx*bp_frames[:,:3,0]-dy*bp_frames[:,:3,1]+dz*bp_frames[:,:3,2]
        p7=bp_frames[:,3,:3]+dx*bp_frames[:,:3,0]+dy*bp_frames[:,:3,1]-dz*bp_frames[:,:3,2]
        p8=bp_frames[:,3,:3]+dx*bp_frames[:,:3,0]+dy*bp_frames[:,:3,1]+dz*bp_frames[:,:3,2]

        coords=np.hstack((p1,p2,p3,p2,p3,p4,
                          p5,p6,p7,p6,p7,p8,
                          p1,p2,p5,p2,p5,p6,
                          p3,p4,p7,p4,p7,p8,
                          p1,p3,p5,p3,p5,p7,
                          p2,p4,p6,p4,p6,p8)).flatten()
        color=[1,1,0]*(coords.size//3)
        view.shape.add_mesh(coords.tolist(), color, )
        
    view.camera='orthographic'
    view._remote_call('setSize',target='Widget',args=['800px','600px'])
    return view