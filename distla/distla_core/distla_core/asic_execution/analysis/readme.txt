These folders contain setups to perform benchmarking and error analysis of
various DistlaCore functions.

To use:

1) cd into the desired folder (e.g. cd summa/benchmark).
2) run asic_execution/scripts/make_execution_directories.py, configured with
   your desired arguments as documented in the --help string. This creates
   subdirectores v_2/, v_3/, etc, each of which contains the appropriate
   configurations to run an analysis on the eponymous slice with tp.
3) If desired, edit the configuration variables in main.py to reflect the
   parameters you wish to study.
4) cd into the desired ASIC directory, ./submit.sh. This will create a ASIC VM,
   run the user code, copy the resulting data back to the ASIC slice
   directory, and delete the VM. Alternatively you can perform these steps
   manually (cat submit.sh to see them).
5) Each run results in .csv file storing times, TFLOP/s, etc for the
   benchmark.
