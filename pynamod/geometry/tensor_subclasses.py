import torch


class mod_Tensor(torch.Tensor):
    geom_class = None

    def __new__(cls, x, geom_class, *args, **kwargs):
        # Do not use Tensor.__new__(cls, tensor); it triggers torch.tensor(tensor) UserWarning.
        # See https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
        t = torch.as_tensor(x)
        if isinstance(x, torch.Tensor):
            t = t.clone().detach()
        return t.as_subclass(cls)

    def __init__(self, x, geom_class, *args, **kwards):
        self.geom_class = geom_class

    def __getitem__(self, sl):
        item = super().__getitem__(sl)
        item.geom_class = self.geom_class
        return item

    def __setitem__(self, sl, value):

        super().__setitem__(sl, value)
        if self.geom_class and self.geom_class._auto_rebuild_sw:
            is_traj = (self.dim() == 3 and self.shape[-1] == 6) or self.dim() > 3
            full_frame_changed = (isinstance(sl, int) or isinstance(sl, torch.Tensor)) and is_traj
            if isinstance(sl, tuple):
                sl = sl[0]

            if isinstance(sl, slice) or full_frame_changed:
                if self.shape[-1] == 3 or self.shape[-1] == 4:
                    self.geom_class.rebuild('rebuild_local_params')
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rebuild_ref_frames_and_ori')

            elif isinstance(sl, int) or (isinstance(sl, torch.Tensor) and sl.shape == tuple()):
                if self.shape[-1] == 3 or self.shape[-1] == 4:
                    self.geom_class.rebuild('rebuild_local_params', start_index=sl)
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rotate_ref_frames_and_ori', sl)