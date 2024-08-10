import sys
import time

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets import CBool

from .dist_backend import get_dist_backend
from .dist_backend import get_local_rank
from .dist_backend import get_local_world_size

__all__ = ["embed"]


class MultiProcessShellEmbed(InteractiveShellEmbed):
    _local_rank = None
    _dist = None
    _local_world_size = None
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
        if self._local_rank is None:
            self._local_rank = get_local_rank()
            self._dist = get_dist_backend()
            self._local_world_size = get_local_world_size()
        super().mainloop(local_ns, module, stack_depth, compile_flags)

    def cleanup(self):
        self._dist.finish()

    def interact(self):
        self.keep_running = True
        if self._dist.is_dummy:
            if self._local_rank != 0:
                return
            else:
                print("Distributed environment not initialized, only activating Rank 0")

        while self.keep_running:
            if self._dist:
                self._active_rank = self._dist.get()
                if self._active_rank == -1:
                    self.cleanup()
                    return

            if self._local_rank == self._active_rank:
                print(self.separate_in, end="")
                try:
                    code = self.prompt_for_code()
                except EOFError:
                    if (not self.confirm_exit) or self.ask_yes_no(
                        "Do you really want to exit ([y]/n)?",
                        "y",
                        "n",
                    ):
                        self.ask_exit()
                        if not self.keep_running:
                            self.cleanup()
                        continue
                # print(f"Get code from rank {self._local_rank}: {code}")
                if code.startswith("%switch "):
                    target_active = code[7:].strip()
                    if target_active.isdigit():
                        target_active = int(target_active)
                        if target_active >= 0 and target_active < self._local_world_size:
                            print(f"Switching to Rank {target_active}")
                            self._dist.set(target_active)
                            print(f"New active Rank: {self._dist.get()}")
                        else:
                            print(f"Invalid rank id: {target_active}")
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
    shell = MultiProcessShellEmbed.instance(
        _init_location_id="{}:{}".format(
            frame.f_code.co_filename,
            frame.f_lineno,
        ),
        **kwargs,
    )
    shell(
        header="",
        stack_depth=2,
        compile_flags=None,
        _call_location_id="{}:{}".format(
            frame.f_code.co_filename,
            frame.f_lineno,
        ),
    )
    MultiProcessShellEmbed.clear_instance()
