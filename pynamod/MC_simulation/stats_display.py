import time
from tqdm.auto import tqdm

class _Stats_Display:
    def __init__(self,max_steps,mute):
        self._prev_time = time.localtime()
        self._mute = mute
        self._total_step_bar = tqdm(total=max_steps,desc='Steps',disable=mute)
        self._info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        print("Start time:",time.strftime('%D %H:%M:%S',self._prev_time))

    def show_step_data(self,accepted_steps,total_steps,prev_e):
        self.acceptance_rate = 100*accepted_steps/total_steps
        if self._mute:
            cur_time = time.localtime()
            if (cur_time.tm_min - self._prev_time.tm_min) == 5:
                self._prev_time = cur_time
                print(f'Accepted steps: {accepted_steps}, acceptance rate: {self.acceptance_rate:.2f}%')
        else:
            self._total_step_bar.update(1)
            self._info_pbar.set_postfix(E=f'{prev_e:.2f}, Acc.st {accepted_steps}')
            self._info_pbar.set_description(f'Acceptance rate %')

            self._info_pbar.update(self.acceptance_rate-self._info_pbar.n)

    def show_final_data(self,target_accepted_steps,stopped,accepted_steps,total_steps):
        if accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        elif stopped:
            print('interrupted!')
        print("Finish time:",time.strftime('%D %H:%M:%S',time.localtime()))
        if not self._mute:
            print('it/s:', '%.2f'%self._total_step_bar.format_dict["rate"])
        print('accepted steps:',accepted_steps)
        print('total steps:',total_steps)
        print(f'acceptance rate: {self.acceptance_rate:.2f}%')