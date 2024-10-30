# UW Quantum Defect Lab Utilities (base)
This repository contains the base software tools and applications for interfacing
with hardware used in the [Quantum Defect Laboratory](https://sites.google.com/uw.edu/optospintronics-lab/home) 
at the University of Washington.
This repository was initially forked from [`qt3-utils`](https://github.com/qt3uw/qt3-utils) 
(developed for the Quantum Technologies Teaching and Test-Bed (QT3) lab,
also at the University of Washington).

Currently, `qdl-utils` supports the following hardware for experiments with 
solid-state optical emitters and spin-qubits:

* TTL pulsers
  * Quantum Composer Sapphire
  * Spin Core PulseBlaster
* Excelitas SPCM for photon detection
* NI-DAQ card (PCIx 6363) for data acquisition and control
  * Jena System's piezo actuator stage control amplifier
  * Mad City Labs piezo actuator stage control
* Newport Micrometers via serial connection

Additionally, several fully interfaced, Python applications are provided for 
standard and commonly used experiments:

* `qt3scope`: Real-time oscilloscope readout from the SPCM via the digital input 
  terminal on the NI-DAQ board.
* `qt3scan`: Reconfigurable 1-d and 2-d confocal scan imaging.
* `qt3pb`: A simple GUI for controlling the pulse blaster.
* `qdlple`: Reconfigurable resonant excitation (photoluminescence excitation)
  spectroscopy. 
* `qdlmove`: Customizable graphical interface for position control, reconfigurable 
  for mutltiple positioners.
  

Applications are generally structured to be easily configured for different
setups (using supported hardware) via the use of YAML files.
Modifications to include additional hardware should also be relatively straightforward.
Finally, in the case where a custom one-off, experiment is required, many of the controllers 
and hardware interfaces can be utilized directly without the development of a GUI
or application.

### Intended usage
Currently this repository is still under development to add generic, widely used
functionality.
Once these features have been added, this repository will no longer be in active 
development.
Instead, users are expected to fork from this repository and actively maintain their forks
with their own applications and hardware.
In the event that specific features, changes, or bug fixes are of generic interest to multiple 
users, this repository may be merged/updated via a Pull Request.


## Setup

### Prerequisites

The utilities in this package depend on publicly available Python packages found
on PyPI and drivers built by National Instruments for DAQmx and
SpinCore for the PulseBlaster. These libraries must be installed separately.

* [National Instruments DAQmx](https://nidaqmx-python.readthedocs.io/en/latest/)
  * [driver downloads](http://www.ni.com/downloads/)
* [SpinCore's PulseBlaster](https://www.spincore.com/pulseblaster.html)
  * [spinAPI driver](http://www.spincore.com/support/spinapi/)

### Installation

Because this package is intended to be forked and customized for use in a
variety of experimental setups, we do not intend to release this package on PyPI.
Instead, users must clone `qdl-utils` (or their specific fork) onto their
own machine and install it locally.
Instructions for how to do this are provided below.

Note that Pull Requests (and commits to the `main` branch) on `qdl-utils` 
will generally be rejected.
Thus, it is strongly discouraged to install `qdl-utils` directly if you 
intend on tracking your changes with GitHub.
Instructions for creating a fork are provided below.


#### 0. Create your fork of the `qdl-utils` repository

Follow [these instructions to create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) 
via the GitHub web browser.
For QDL members, create the fork to be owned by the UW-Quantum-Defect-Lab GitHub organization
and name the fork appropriately as to be distinguishable for your project, e.g. `qdl-utils-diamond`
or `qdl-utils-magpi`.
For the purposes of this tutorial we will assume that the fork is named `my-qdl-utils-fork`.

Once the fork has been created on GitHub, move on to the next step.
Do not clone your fork yet; we will do this in the next steps.

If you wish to completely decouple from `qdl-utils` then you can follow [these
instructions to detach your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/detaching-a-fork).
Note that this will prevent you from being able to Pull Request `qdl-utils` or
straightforwardly merge updates onto your repository.

#### 1. Create a development environment 

It is highly recommended that you utilize some form of virtual environment to
install your fork `my-qdl-utils-fork` (in this example).
It is assumed that you will use [Anaconda](https://docs.anaconda.com/anaconda/install/)
or its lightweight version [Miniconda](https://docs.anaconda.com/miniconda/).
If you are unfamiliar with `conda` please review [this tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).
For Windows users new to Anaconda, it is recommended to install Anaconda for only your user
and then utilize the provided Anaconda PowerShell Prompt terminal for the remainder
of this process.

Create a new virtual environment with Python 3.9. In this example we will name the
virtual environment `qdlutils`, but you can change it to something else if desired.
In a terminal configured to for `conda`, run

```
> conda create --name qdlutils python=3.9
```

to create the virtual environment and then activate it via

```
> conda activate qdlutils
```


#### 2. Clone your fork

Navigate (in the terminal) to the directory in which you would like to install
the full `qdl-utils` repository.
This includes not only the `qdlutils` package source code, but also examples, 
and any other files stored in the repository.
Note that you will often need to edit the source code so pick a directory which
you can easily access.

Once your present working directory is the desired location, clone your fork via

```
> git clone <URL of my-qdl-utils-fork>
```

You can get the URL for your fork from its GitHub page by clicking on the green 
"Code" button as shown in the following picture, then copying the URL:

![repo_url](https://github.com/user-attachments/assets/0396ff47-59e1-47fa-a7d9-053b81d58298)

For example, if we wanted to install the `qdl-utils` repo itself we could run
`git clone https://github.com/UW-Quantum-Defect-Lab/qdl-utils.git`.
Your URL will probably be something like `git clone https://github.com/UW-Quantum-Defect-Lab/my-qdl-utils-fork.git`.
Note that, for reasons explained above, cloning `qdl-utils` directly is not
recommended.


#### 3. Install the local repository in "editor" mode

Finally, move into the the newly created clone of your repository.
Assuming that the name of your fork is `my-qdl-utils-fork` we first move into 
the repository by

```
> cd my-qdl-utils-fork
```

and then install it locally in editor mode via

```
> pip install -e . 
```

Do not forget the `.`! 
This refers `pip` to the `pyproject.toml` file in the home directory
of the repository that does the installation.
The `-e` option ensures that any edits you make to the local repository 
are reflected in subsequent imports in Python scripts (without this you
would have to `pip` install after each edit).


Finally confirm that the installation worked correctly by launching the
Python interpreter and then importing the package:

```
> Python
Python 3.9 ...
>>> import qdlutils
>>>
```

#### 4. (Optional) Update Tk/Tcl

Upgrading Tcl/Tk via Anaconda overcomes some GUI bugs on Mac OS Sonoma

```
conda install 'tk>=8.6.13'
```

# LICENSE

[LICENCE](LICENSE)
