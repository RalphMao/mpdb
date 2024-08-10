import sys
import time
from traitlets import Bool, CBool, Unicode
from IPython.terminal.embed import InteractiveShellEmbed

from .dist_backend import get_local_rank, get_dist_backend

   
    
class MultiProcessShellEmbed(InteractiveShellEmbed):
    _local_rank = get_local_rank()
    _dist = get_dist_backend()
    _active_rank = 0
    display_banner = CBool(False)
    exit_msg = None

    def mainloop(
        self,
        local_ns=None,
        module=None,
        stack_depth=0,
        compile_flags=None,
    ):
        # print("Try to enter mainloop")
        super().mainloop(local_ns, module, stack_depth, compile_flags)

    def cleanup(self):
        n_sec = 3
        self._dist.finish()
        print(f"Sending message to other sessions. Exiting in {n_sec} seconds.")
        time.sleep(3)


    def interact(self):
        self.keep_running = True
        if self._local_rank != 0 and self._dist.is_dummy:
            # print(f"Rank {self._local_rank} exiting, as distributed backend is not found")
            return

        while self.keep_running:

            if self._dist:
                self._active_rank = self._dist.get()
                if self._active_rank == -1:
                    return

            if self._local_rank == self._active_rank:
                print(self.separate_in, end='')
                try:
                    code = self.prompt_for_code()
                except EOFError:
                    if (not self.confirm_exit) \
                            or self.ask_yes_no('Do you really want to exit ([y]/n)?','y','n'):
                        self.ask_exit()
                        if not self.keep_running:
                            self._dist.set(-1)
                        continue
                # print(f"Get code from rank {self._local_rank}: {code}")
                if code.startswith("%mpdb switch "):
                    target_active = code[13:].strip()
                    if target_active.isdigit():
                        target_active = int(target_active)
                        print(f"Switching to Rank {target_active}")
                        self._dist.set(target_active)
                        print(f"New active Rank: {self._dist.get()}")
                    else:
                        print("Invalid command")
                    continue

                if code:
                    self.run_cell(code, store_history=True)
                    if not self.keep_running:
                        self.cleanup()

            else:
                time.sleep(0.5)

def embed(**kwargs):
    frame = sys._getframe(1)
    shell = MultiProcessShellEmbed.instance(_init_location_id='%s:%s' % (
        frame.f_code.co_filename, frame.f_lineno), **kwargs)
    shell(header="", stack_depth=2, compile_flags=None,
        _call_location_id='%s:%s' % (frame.f_code.co_filename, frame.f_lineno))
    MultiProcessShellEmbed.clear_instance()
