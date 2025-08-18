## Before running the app, download the model file from huggingface and place it in the app directory:

```
wget https://huggingface.co/stoeckle/extrusion-unet/resolve/main/flownet_model.keras -O flownet_model.keras
```
## Installing dependencies

The app requires FEniCSx v0.9, Tensorflow (>2.18), and a number of Python libraries for data processing. To maintain compatbility with the deep learning model file itself, Python 3.12 should be used. A conda environment is recommended to ensure package compatibility:

```
conda create -n fenicsx_tf_env python=3.12
conda activate fenicsx_tf_env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install gmsh scipy scikit-image rdp shapely tensorflow 
```
