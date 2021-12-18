# tp
This is a utility for interacting with and executing code on Cloud ASIC VM
instances. This utility is designed to simplify the process of code execution
on both asic_nodes and asic_clusters.

In addition to providing a CLI, all functionality can be accessed
programatically to build custom orchestration scripts.


## Usage
pip install path/to/tp

```
$ tp
	# help

$ tp info --help
	# cmd help

$ tp info
	# tp will check for a configuration yaml in the cwd.

$ tp info -f path/to/asic.yaml
	# A different yaml can be supplied

$ tp ssh --name 'my-asic'
	# Any supplied values will be overwritten

$ tp run arg1 arg2 -name 'my-asic' -entry_point 'my_python_file.py'
```

Look at the included example `asic.yaml` for a list of yaml configurable options
and user scripts.

## Specifying runtime ENV variables
Environment variables can be passed to the `run` function either through the
config file in yaml format:

```yaml
run_env:
	MY_ENV_VAR: abc
	OTHER_VAR: 123
```

or as a CLI argument:

`tp run --run_env='{"MY_ENV_VAR": "abc"}'`


## SSH Configuration
`tp` will looks for ssh keys at the default gcloud ssh key location:
`~/.ssh/google_compute_engine`

Because asic hosts can reboot and change their keys all ssh connections will
by default ignore known_hosts files completely.

In the future an option to reference the default gcloud known hosts file may 
be added:
`~/.ssh/google_compute_known_hosts`


## Accessing ASIC Host Information
The name of the ASIC instance and the worker number are made available to python
through ENV variables:

`TP_ASIC_NAME`

`TP_ASIC_WORKER`


## Changing Python version
Until a more more engineered solution is in place, python version can be changed
by adding something like the following to the user preflight script:
```shell
sudo apt install python3.9 -y
python3.9 -m pip install pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
pip install setuptools
pip install requests
```

## ASIC Logging
ASIC logs are written to `/tmp/asic_logs` with the latest log files available
at symlinks of `asic_driver.{LOG_LEVEL}`

ASIC logging is controlled by the environment variable `ASIC_MIN_LOG_LEVEL`.
To generate asic log files set this to the most verboselog level (0).

ASIC logs can be generated on stderr and sent to the terminal by specifying the
stderr log level variable `ASIC_STDERR_LOG_LEVEL`.

## Profiling
TP includes a command to start a tensorboard server to help with profiling jax
applications. See JAX documentation for examples 
[here](https://jax.readthedocs.io/en/latest/profiling.html#programmatic-capture).

## Feature Wish List
Forthcoming features:
* Output stream filtering
* env vars passed to every remote execution scope
* Multiple dist folder support
* Add builtin support for changing VM python version with pyenv
* Add testing

These are things I hope to implement soon:
* More polished error messages and arg validation
* More polished help menus
* Terminal completion
* Re-use single SSH socket for all communication in 'run'
* More built-in data processing utilities
* pretty colors and shiny loading bars
