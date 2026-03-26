import signal

import numpy as np
import torch

from pynamod.geometry.trajectories import H5_Trajectory, Tensor_Trajectory
from pynamod.MC_simulation.rotation_handler import _Rotation_Handler
from pynamod.MC_simulation.stats_display import _Stats_Display


class Iterator:
    def __init__(self, cg_structure, energy, sigma_transl=1, sigma_rot=1):
        self.cg_structure = cg_structure
        self.energy = energy
        self._rotation_handler = _Rotation_Handler(sigma_transl, sigma_rot)

        self.res_trajectory = cg_structure.dna.geom_params.trajectory

    def run(self, target_accepted_steps=int(1e5), max_steps=int(1e6), transfer_to_memory_every=None, save_every=1,
            mute=False, KT_factor=1, integration_mod='minimize', device='cpu', traj_init_step=None):

        self._prepare_system(target_accepted_steps, transfer_to_memory_every, device, integration_mod, traj_init_step, KT_factor)

        self._stop_loop = False
        signal.signal(signal.SIGINT, self._signal_handler)

        self._stats_display = _Stats_Display(max_steps, mute)

        while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:

            self._integration_step(save_every, integration_mod)

            self._stats_display.show_step_data(self.accepted_steps, self.total_step, self.prev_e.sum().item())
            # Stop only when a step is completed in case of keyboard interrupt.
            if self._stop_loop:
                break

        self._transfer_to_memory(steps=self.trajectory.cur_step)
        self._stats_display.show_final_data(target_accepted_steps, self._stop_loop, self.accepted_steps, self.total_step)

    def to(self, device):
        self.trajectory.to(device)
        self.energy.to(device)
        self._rotation_handler.to(device)

    def _prepare_system(self, target_accepted_steps, transfer_to_memory_every, device, integration_mod, traj_init_step, KT_factor):
        if not transfer_to_memory_every:
            transfer_to_memory_every = target_accepted_steps
        self.transfer_to_memory_every = transfer_to_memory_every
        self.total_step = self.accepted_steps = self.last_accepted = 0
        if not traj_init_step:
            traj_init_step = len(self.res_trajectory) - 1
        cur_step = self.res_trajectory.cur_step
        self.res_trajectory.cur_step = traj_init_step
        self._create_tens_trajectory()
        self.to(device)

        if 'energies' not in self.res_trajectory.attrs_names:
            self.res_trajectory.add_attr('energies', (4,))
        self.prev_e = torch.stack(self.energy.get_energy_components(self.trajectory))
        self.res_trajectory._set_frame_attr('energies', self.prev_e.cpu())
        self.energy_comp_traj = torch.zeros(self.transfer_to_memory_every, 4, device=device)

        self.res_trajectory.cur_step = cur_step

        self._set_change_indices(integration_mod)

        self._scaled_KT = KT_factor*self.energy.KT

    def _set_change_indices(self, integration_mod):
        self.movable_ind = torch.arange(self.trajectory.data_len, dtype=int)[self.cg_structure.dna.movable_steps]
        if self.movable_ind[0] == 0:
            self.movable_ind = self.movable_ind[1:]
        if integration_mod == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            self.movable_ind_len = self.movable_ind.shape[0]

    def _create_tens_trajectory(self):
        init_local_params = self.cg_structure.dna.geom_params.local_params
        init_ref_frames = self.cg_structure.dna.geom_params.ref_frames
        init_ori = self.cg_structure.dna.geom_params.origins

        if self.cg_structure.proteins:
            init_prot_ori = torch.vstack([
                p.origins.detach().clone() if isinstance(p.origins, torch.Tensor) else torch.as_tensor(p.origins)
                for p in self.cg_structure.proteins
            ])
        else:
            init_prot_ori = torch.empty((0, 1, 3))

        ln = init_ref_frames.shape[0]
        traj_len = self.transfer_to_memory_every+1
        dtype = init_ref_frames.dtype
        prot_origins_ln = init_prot_ori.shape[0]

        self.trajectory = Tensor_Trajectory(dtype, traj_len, ln, torch.tensor, attrs_names=['prot_origins'], shapes=[(traj_len, init_prot_ori.shape[0], 1, 3)])
        self.trajectory.origins, self.trajectory.ref_frames = init_ori, init_ref_frames
        self.trajectory.local_params = init_local_params
        self.trajectory.prot_origins = init_prot_ori

    def _integration_step(self, save_every, integration_mod):

        change_indices = self._get_cur_change_index(integration_mod)
        self._rotation_handler.apply_rotation(change_indices, self.trajectory)

        e_dif_components, e_mat, s_mat = self.energy.get_energy_dif(self._rotation_handler, change_indices[1], self.prev_e)

        e_dif_components = torch.stack(e_dif_components)
        cur_e = torch.stack(self.energy.get_energy_components(self._rotation_handler, save_matr=False))
        # d = ((cur_e - self.prev_e)[1:3] - e_dif_components[1:3])
        # if abs(d[0]) > 10**-4 or abs(d[1]) > 10**-4:
        #     print(d)

        Del_E = e_dif_components.sum()

        r = torch.rand(1).item()
        self.total_step += 1

        if not Del_E.isnan() and Del_E < 0 or (not (torch.isinf(torch.exp(Del_E))) and (r <= torch.exp(-Del_E/self._scaled_KT))):
            self.energy.update_matrices(e_mat, s_mat, change_indices[1])
            self.prev_e += e_dif_components
            self.accepted_steps += 1

            self._rotation_handler.set_new_traj_params(self.trajectory)

            if (self.accepted_steps % save_every) == 0:
                self.energy_comp_traj[self.trajectory.cur_step] = self.prev_e
                self.trajectory.cur_step += 1
                if self.trajectory.cur_step == self.transfer_to_memory_every:
                    self.prev_e = torch.hstack(self.energy.get_energy_components(self._rotation_handler))
                    self._transfer_to_memory()

                    self.trajectory.cur_step = 0

                self._rotation_handler.set_new_traj_params(self.trajectory)

    def _signal_handler(self, signum, frame):
        self._stop_loop = True

    def _get_cur_change_index(self, integration_mod):
        if integration_mod == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            cur_index = torch.randint(self.movable_ind_len, (1,))

        dna_change_index = self.movable_ind[cur_index]
        prot_change_index = sum([protein.n_cg_beads for protein in self.cg_structure.proteins if protein.ref_pair.ind < dna_change_index])
        return dna_change_index, prot_change_index

    def _transfer_to_memory(self, steps=None):
        if steps is None:
            steps = self.transfer_to_memory_every

        dna_len = self.trajectory.data_len
        origins_traj = self.trajectory.origins_traj[:steps, :dna_len].numpy(force=True)
        ref_frames_traj = self.trajectory.ref_frames_traj[:steps].numpy(force=True)
        local_params_traj = self.trajectory.local_params_traj[:steps].numpy(force=True)
        energy_comp_traj = self.energy_comp_traj[:steps].numpy(force=True)

        for i in range(steps):
            self.res_trajectory.cur_step += 1
            self.res_trajectory.add_frame(self.res_trajectory.cur_step, origins=origins_traj[i], ref_frames=ref_frames_traj[i],
                                          local_params=local_params_traj[i], energies=energy_comp_traj[i])
