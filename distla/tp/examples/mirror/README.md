## Mirroring

This example asic.yaml demonstrates tp's mirroring.
In this particular example the distla_core repo is selected in the asic.yaml
as the dist_dir and will be mirrored to the default destination on the ASIC VMs.

Run `tp create` and then `tp mirror` in this directory to continuously mirror
the distla_core repo to a ASIC VM.

Another directory could be selected for mirroring by specifying as dist_dir in
the asic.yaml or overwriting in the cli with `tp mirror --dist_dir=./somewhere`.