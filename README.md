# ElasticRods
![](https://lgg.epfl.ch/publications/2019/XShells/project_teaser.png)

A simulation framework for [discrete elastic rods](http://www.cs.columbia.edu/cg/threads/) and X-Shells written in C++ with Python bindings.
This is the codebase for our 2019 Siggraph paper, [X-Shells](http://julianpanetta.com/publication/xshells/).
Check out the section on [reproducing the paper
figures](#reproducing-the-paper-figures) for pointers to some Jupyter notebooks to
try.

# Getting Started
## C++ Code Dependencies

The C++ code relies on `Boost` and `CHOLMOD/UMFPACK`, which must be installed
separately. Some parts of the code (actuator sparsification, design
optimization) also depend on the commercial optimization package
[`Knitro`](https://www.artelys.com/solvers/knitro/); these will be omitted from
the build if `Knitro` is not found.

The code also relies on several dependencies that are included as submodules:
[MeshFEM](https://github.com/MeshFEM/MeshFEM),
[libigl](https://github.com/libigl/libigl),
[spectra](https://github.com/yixuan/spectra), and 
[visvalingam_simplify](https://github.com/shortsleeves/visvalingam_simplify).

### macOS
You can install all the mandatory dependencies on macOS with [MacPorts](https://www.macports.org). When installing SuiteSparse, be sure to get a version linked against `Accelerate.framework` rather than `OpenBLAS`; on MacPorts this is achieved by requesting the `accelerate` variant, which is no longer the default.

```bash
# Build/version control tools, C++ code dependencies
sudo port install cmake boost ninja
sudo port install SuiteSparse +accelerate
# Install nodejs/npm using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 17 && nvm use 17
```

### Ubuntu 19.04
A few more packages need to be installed on a fresh Ubuntu 19.04 install:
```bash
# Build/version control tools
sudo apt install git cmake ninja-build
# Dependencies for C++ code
sudo apt install libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libsuitesparse-dev
# LibIGL/GLFW dependencies
sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
# Dependencies (pybind11, jupyterlab/notebooks)
sudo apt install python3-pip npm
# Ubuntu 19.04 packages an older version of npm that is incompatible with its nodejs version...
sudo npm install npm@latest -g
```

## Obtaining and Building

Clone this repository *recursively* so that its submodules are also downloaded:

```bash
git clone --recursive https://github.com/jpanetta/ElasticRods
```

Build the C++ code and its Python bindings using `cmake` and your favorite
build system. For example, with [`ninja`](https://ninja-build.org):

```bash
cd ElasticRods
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Running the Jupyter Notebooks

The preferred way to interact with the rods code is in a Jupyter notebook,
using the Python bindings.

### JuptyerLab and Extensions
To run the Jupyter notebooks, you will need to install JupyterLab and
[my fork](https://github.com/jpanetta/pythreejs) of the `pythreejs` library.
JupyterLab can be installed through `pip`, and the following commands should
set up all the requirements on both macOS and Ubuntu. Using a virtual environment
is strongly recommended!

```bash
pip3 install wheel # Needed if installing in a virtual environment
# Recent versions of jupyterlab and related packages cause problems:
#   JupyerLab 3.4 and later has a bug where the tab and status bar GUI
#                 remains visible after taking a viewer fullscreen
#   ipykernel > 5.5.5 clutters the notebook with stdout content
#   ipywidgets 8 and juptyerlab-widgets 3.0 break pythreejs
pip3 install jupyterlab==3.3.4 ipykernel==5.5.5 ipywidgets==7.7.2 jupyterlab-widgets==1.1.1
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .

pip3 install matplotlib scipy
```

You may need to add the following to your shell startup script for the installation of `pythreejs`'s dependencies during `pip3 install -e .` to succeed:
```
export NODE_OPTIONS=--openssl-legacy-provider;
```

Launch Jupyter lab from the root python directory:
```bash
cd python
jupyter lab
```

Now try opening and running an example notebook, e.g.,
[`python/Demos/MantaRayDemo.ipynb`](https://github.com/jpanetta/ElasticRods/blob/master/python/Demos/MantaRayDemo.ipynb).
Several other demo notebooks are included in the same directory to introduce
you to some of the features of the codebase.

### Reproducing the Paper Figures
Each figure in the paper that includes an X-Shell simulation has corresponding Jupyter notebook(s) in
[`python/XShellSiggraphPaperFigures`](https://github.com/jpanetta/ElasticRods/tree/master/python/XShellSiggraphPaperFigures).
Note that these notebooks generate the raw data used to produce the figures, but
the final renderings for most figures were produced in Rhino using scripts that
are not included in this release. Basic preview renderings are available in the
in-notebook viewer.
