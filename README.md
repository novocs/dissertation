## Introduction
This repository consists of python interfaces used in a research project on clustering. The relevant code can be found
in the **clusterlib** folder. There are also jupyter notebooks present to describe how these interfaces can be used. 
- EmbeddingClustering.ipynb: provides examples on using the deep clustering process. Generally, this invoves the use of
an autoencoder followed by a shallow clustering algorithm.
- KPrototypes.ipynb: provides an example on using the Kprototype interface.


## Installing Requirements.
The code in this repository was implemented with Python 3.10.6 and open source packages. It should be noted that the package versions 
used in this repository are different from those used in the paper. This is mainly because analysis performed in the 
paper was completed on Amazon Web Services Sagemaker, which prevented installation of the more current verison of some 
packages. It has been verified that using more current versions does not impact the processes developed. Instructions for 
creating a suitable environment for running the code are as follows:
* Clone this repository and enter the root directory. If you have git installed (instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)), running the commands below in the commandline
would help achieve this, otherwise a zip of the repository can simply be downloaded and opened. 
  ```commandline
  git clone https://github.com/novocs/dissertation.git
  cd dissertation
  ```
* Install Python 3.10.6 and create a virtual environment. This can be done using your preferred python and virtual environment manager. 
In producing this code, pyenv and pyenv-virtualenv was used for this step on a macOS.
  * **pyenv**: instructions for installing and using pyenv can be found [here](https://github.com/pyenv/pyenv).
  * **pyenv-virtualenv**: instructions for installing and using pyenv-virtualenv can be found [here](https://github.com/pyenv/pyenv-virtualenv).
* Run the command below to install required packages
    ```commandline
    pip3 install -r requirements.txt
    ```
* Install tensorflow version 2.10.0 using the relevant instructions for your operating system. System specific instructions for this can be found 
[here](https://www.tensorflow.org/install/pip#linux) and may require the installation of Anaconda (conda). For example
  * For macOS, the command below is generally sufficient.
    ```commandline
    pip3 install tensorflow-macos==2.10.0
    ```
  * For Linux systems, conda installation is recommended and instructions for this can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 
  Following this the command below can be ran.
    ```commandline
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    python3 -m pip install tensorflow==2.10.0
    ```

## Running the code
Open any of the jupyter notebooks and begin running the code to familiarize yourself with the interface.

## References
[1] McConville, R., Santos-Rodríguez, R., Piechocki, R., & Craddock, I. (2019). N2D: (Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding. IEEE Xplore. https://doi.org/10.1109/ICPR48806.2021.9413131

[2] Advanced Usage — n2d 0.3.1 documentation. N2d.readthedocs.io. Retrieved July 27 , 2022, from https://n2d.readthedocs.io/en/latest/extending.html