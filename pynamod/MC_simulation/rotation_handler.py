import torch

from pynamod.geometry.bp_step_geometry import Geometry_Functions


class _Rotation_Handler(Geometry_Functions):
    def __init__(self, sigma_transl, sigma_rot):
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot

        self.change = torch.zeros(6, dtype=torch.double)
        self.normal_mean = torch.zeros(6, dtype=torch.double)
        self.normal_scale = torch.tensor([*[self.sigma_transl]*3, *[self.sigma_rot]*3])

    def to(self, device):
        self.change = self.change.to(device)
        self.normal_mean = self.normal_mean.to(device)
        self.normal_scale = self.normal_scale.to(device)

    def apply_rotation(self, change_indices, trajectory):
        self.ref_frames = trajectory.ref_frames.clone()
        self.local_params = trajectory.local_params.clone()
        self.origins = trajectory.origins.clone()
        self.rlsp_origins = trajectory.rlsp_origins.clone()
        self.local_params[change_indices[0]] += torch.normal(mean=self.normal_mean,std=self.normal_scale,out=self.change)

        self.rotate_ref_frames_and_ori(*change_indices)

    def compare(self):
        new = _Rotation_Handler(1, 1)
        new.local_params = self.local_params.clone()
        new.origins = self.origins.clone()
        new.ref_frames = self.ref_frames.clone()
        new.len = new.ref_frames.shape[0]
        new.rebuild_ref_frames_and_ori(start_ref_frame=self.ref_frames[0],
                                       start_origin=self.origins[0])
        return ((new.origins-self.origins).abs().max(),(new.ref_frames-self.ref_frames).abs().max())

    def set_new_traj_params(self, trajectory):

        trajectory.origins = self.origins
        trajectory.ref_frames = self.ref_frames
        trajectory.local_params = self.local_params
        trajectory.rlsp_origins = self.rlsp_origins

