# crystal4D

Package for complex convolution network and spectral pooling used for pixel-wise mapping of diffraction space images

## Installation

The recommended installation for crystal4D uses the Anaconda python distribution.
First, download and install Anaconda. Instructions can be found at www.anaconda.com/download.
Then open a terminal and run

```
conda update conda
conda create -n crystal4D python==3.8
conda activate crystal4D
conda install tensorflow-gpu==2.4.1
conda install pip
pip install tensorflow-addons
pip install crystal4D
```

For mac OS X, please run

```
conda update conda
conda create -n crystal4D python==3.8
conda activate crystal4D
conda install pip
pip install tensorflow==2.4.1
pip install tensorflow-addons
pip install crystal4D
```

In order, these commands
- ensure your installation of anaconda is up-to-date
- make a virtual environment - see below!
- enter the environment
- make sure your new environment talks nicely to pip, a tool for installing Python packages
- use pip to install crystal4D
- on Windows: enable python to talk to the windows API

Please note that virtual environments are used in the instructions above, to make sure packages that have different dependencies don't conflict with one another.
Because these directions install crystal4D to its own virtual environment, each time you want to use crystal4D, you'll need to activate this environment.
You can do this in the command line with `conda activate crystal4D`, or, if you're using the Anaconda Navigator, by clicking on the Environments tab and then clicking on `crystal4D`.
