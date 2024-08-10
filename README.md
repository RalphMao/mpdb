# IPython for Multi-Process Debugging

IPython is great for debugging but using it in multi-process environment is awkward. `mpdb` is designed to make multi-process debugging (esp. PyTorch multi-GPU) less frustrating.

## Installation

 ```
 pip install git+https://github.com/RalphMao/mpdb
 ```

## Usage

 1. Add `mpdb.embed()` (similar to `IPython.embed()`) to the target location of the python script.
 2. Initially only Rank 0 will be activated with a IPython shell.
 3. To switch to other ranks, use line magic `%switch <rank_id>` .



## TODO

 - [ ] Add MPI and Slurm examples
 - [ ] Support multi-process equivalent of `ipdb.set_trace`
 - [ ] More general sync mechanism without torch dependency
 - [ ] Support multi-node debugging on Slurm
