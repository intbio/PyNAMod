import torch
import signal
from pynamod.geometry.trajectories import Tensor_Trajectory
from pynamod.MC_simulation.stats_display import _Stats_Display
from pynamod.MC_simulation.rotation_handler import _Rotation_Handler


class Iterator:
    def __init__(self, cg_structure, energy, sigma_transl=1, sigma_rot=1):
        self.cg_structure = cg_structure
        self.energy = energy
        self._rotation_handler = _Rotation_Handler(sigma_transl, sigma_rot)

        self.res_trajectory = cg_structure.dna.geom_params.trajectory

    def run(self, target_accepted_steps=int(1e5), max_steps=int(1e6), transfer_to_memory_every=None, save_every=1, rebuild_every=400,
            KT_factor=1, mute=False, integration_mode='minimize', device='cpu', traj_init_step=None, dtype=torch.float,debug = False,output=None):
        '''Starts Monte Carlo Simulation.
    
            Attributes:
    
            - **target_accepted_steps** - number of accepted simulation steps to reach, default 1e5.
    
            - **max_steps** - maximum number of attempted steps. If reached run will be terminated even if target_accepted_steps was not matched. Default 1e6.
    
            - **transfer_to_memory_every** - When using gpu, accepted frames will be flushed to memory or h5 file (depending on trajectory of CG structure) each chosen number of accepted steps. If None will flush only at the end of the simulation. Default None.
    
            - **save_every** - Accepted frames will be saved in resulting trajectory every chosen number of accepted steps. Default 1.

            - **rebuild_every** - Full procedure of rebuilding from bp step paramaters will be used every chosen number of accepted steps instead of simplified rotation and shift. This is necessary to negate computing error. The required frequency can be determined using debug (see debug attribute). Default 400.

            - **KT_factor** - factor that will be used for KT from pynamod.Energy object in Metropolis criterion. Defailt 1.

            - **mute** - If False, tqdm bars will be used to show simulation data, if True instead every 5 minutes number of accepted steps and acceptance rate will be printed. Default False.

            - **integration_mode** - 'minimize' or 'random_step'. If minimize, Iterator will go through movable bp steps repeatedly using one of them as a changing bp step. If 'random_step', Iterator will choose a bp step to change randomly from movable bp steps. Default 'minimize'.

            - **device** - pytorch style 'cpu' or 'cuda' ('cuda:n'). Device to use for simulations. Default 'cpu'.

            - **traj_init_step** - If not None, frame from this trajectory step will be used as an initial conformation. If None, current step from trajectory is used. Default None.

            - **dtype** - tensor dtype to use in calculations. Default torch.float.

            - **debug** - 
        '''
        self._prepare_system(target_accepted_steps, transfer_to_memory_every, rebuild_every,
                             device, integration_mode, traj_init_step, KT_factor, dtype, debug)

        self._stop_loop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        self._stats_display = _Stats_Display(max_steps,mute,output)

        while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:

            self._integration_step(save_every, integration_mode)

            self._stats_display.show_step_data(self.accepted_steps, self.total_step, self.prev_e.sum().item())
            #Stop only when a step is completed in case of keyboard interrupt.
            if self._stop_loop:
                break

        self._transfer_to_memory(steps=self.trajectory.cur_step)
        self._stats_display.show_final_data(target_accepted_steps,self._stop_loop,self.accepted_steps,self.total_step)

    def to(self, device):
        self.trajectory.to(device)
        self.energy.to(device)
        self._rotation_handler.to(device)
        for group in self.cg_structure.rlsp_groups:
            group.ref_vectors = group.ref_vectors.to(device)

    def _prepare_system(self, target_accepted_steps, transfer_to_memory_every, rebuild_every,
                        device, integration_mode, traj_init_step, KT_factor, dtype, debug):
        if rebuild_every is None:
            rebuild_every = target_accepted_steps
        self.rebuild_every = rebuild_every
        if debug:
            self.dif_cutoff = 10 ** -6
        self.debug = debug
        if not transfer_to_memory_every:
            transfer_to_memory_every = target_accepted_steps
        self.transfer_to_memory_every = transfer_to_memory_every
        self.total_step = self.accepted_steps = self.last_accepted = 0
        if not traj_init_step:
            traj_init_step = len(self.res_trajectory) - 1
        cur_step = self.res_trajectory.cur_step
        self.res_trajectory.cur_step = traj_init_step
        self._create_tens_trajectory(dtype)
        if 'rlsp_origins' not in self.res_trajectory.attrs_names:
            self.res_trajectory.add_attr('rlsp_origins',self.trajectory.rlsp_origins.shape)
            self.res_trajectory.rlsp_origins = self.trajectory.rlsp_origins
        self.to(device)

        if 'energies' not in self.res_trajectory.attrs_names:
            self.res_trajectory.add_attr('energies', (4,))
        self.prev_e = torch.stack(self.energy.get_energy_components(self.trajectory))
        self.res_trajectory._set_frame_attr('energies', self.prev_e.cpu())
        self.energy_comp_traj = torch.zeros(self.transfer_to_memory_every, 4, device=device)

        self.res_trajectory.cur_step = cur_step

        self._set_change_indices(integration_mode)

        self._scaled_KT = KT_factor*self.energy.KT

    def _set_change_indices(self, integration_mode):
        self.movable_ind = torch.arange(self.trajectory.data_len, dtype=int)[self.cg_structure.dna.movable_steps]
        if self.movable_ind[0] == 0:
            self.movable_ind = self.movable_ind[1:]
        if integration_mode == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mode == 'random_step':
            self.movable_ind_len = self.movable_ind.shape[0]

    def _create_tens_trajectory(self, dtype):
        init_local_params = self.cg_structure.dna.geom_params.local_params.to(dtype)
        init_ref_frames = self.cg_structure.dna.geom_params.ref_frames.to(dtype)
        init_ori = self.cg_structure.dna.geom_params.origins.to(dtype)
        init_rlsp_ori = self.cg_structure.origins.to(dtype)

        ln = init_ref_frames.shape[0]
        traj_len = self.transfer_to_memory_every + 1
        dtype = init_ref_frames.dtype

        self.trajectory = Tensor_Trajectory(dtype,traj_len,ln,torch.tensor,attrs_names=['rlsp_origins'],shapes=[(traj_len,init_rlsp_ori.shape[0],1,3)])
        self.trajectory.origins, self.trajectory.ref_frames = init_ori, init_ref_frames
        self.trajectory.local_params = init_local_params
        self.trajectory.rlsp_origins = init_rlsp_ori

    def _integration_step(self, save_every, integration_mode):

        change_indices = self._get_cur_change_index(integration_mode)
        self._rotation_handler.apply_rotation(change_indices, self.trajectory)

        e_dif_components, e_mat, s_mat = self.energy.get_energy_dif(self._rotation_handler, change_indices[1], self.prev_e)
        e_dif_components = torch.stack(e_dif_components)
        if self.debug:
            cur_e = torch.stack(self.energy.get_energy_components(self._rotation_handler, save_matr=False))
            energy_error = ((cur_e - self.prev_e) - e_dif_components)
            if sum(abs(energy_error) > 0.01) > 0:
                print(f'Energy difference error is {energy_error}')
            pos_dif = self._rotation_handler.compare()
            if pos_dif[0] > self.dif_cutoff or pos_dif[1] > self.dif_cutoff:
                print(f'Origins error has reached {pos_dif[0]:6f} and ref_frames error has reached {pos_dif[1]:6f} at accepted step {self.accepted_steps}')
                self.dif_cutoff *= 10

        Del_E = e_dif_components.sum()
        r = torch.rand(1).item()
        self.total_step += 1

        if not Del_E.isnan() and Del_E < 0 or (not (torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/self._scaled_KT))):
            self.energy.update_matrices(e_mat, s_mat, change_indices[1])
            self.prev_e += e_dif_components
            self.accepted_steps += 1

            if self.accepted_steps % self.rebuild_every == 0:
                self._rotation_handler.len = self._rotation_handler.origins.shape[0]
                self._rotation_handler.rebuild_ref_frames_and_ori(
                    start_ref_frame=self._rotation_handler.ref_frames[0],
                    start_origin=self._rotation_handler.origins[0])

                rlsp_origins = []
                for rlsp_group in self.cg_structure.rlsp_groups:
                    ref_ind = rlsp_group.ref_pair.ind
                    ref_r = self._rotation_handler.ref_frames[ref_ind]
                    ref_ori = self._rotation_handler.origins[ref_ind]
                    rlsp_origins.append(rlsp_group.get_true_pos(ref_om=ref_ori, ref_Rm=ref_r))

                self.rlsp_origins = torch.vstack(rlsp_origins)

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

    def _get_cur_change_index(self, integration_mode):
        if integration_mode == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mode == 'random_step':
            cur_index = torch.randint(self.movable_ind_len, (1,))

        ref_ori_change_index = self.movable_ind[cur_index]
        rlsp_change_index = sum([group.n_cg_beads for group in self.cg_structure.rlsp_groups if group.ref_pair.ind < ref_ori_change_index])
        return ref_ori_change_index,rlsp_change_index

    def _transfer_to_memory(self, steps=None):
        if steps is None:
            steps = self.transfer_to_memory_every

        dna_len = self.trajectory.data_len
        origins_traj = self.trajectory.origins_traj[:steps, :dna_len].numpy(force=True)
        ref_frames_traj = self.trajectory.ref_frames_traj[:steps].numpy(force=True)
        local_params_traj = self.trajectory.local_params_traj[:steps].numpy(force=True)
        rlsp_origins_traj = self.trajectory.rlsp_origins_traj[:steps].numpy(force=True)
        energy_comp_traj = self.energy_comp_traj[:steps].numpy(force=True)

        for i in range(steps):
            self.res_trajectory.cur_step += 1
            self.res_trajectory.add_frame(self.res_trajectory.cur_step, origins=origins_traj[i],
                                          ref_frames=ref_frames_traj[i], local_params=local_params_traj[i],
                                          rlsp_origins=rlsp_origins_traj[i], energies=energy_comp_traj[i])