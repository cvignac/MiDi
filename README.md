# MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation

Clément Vignac, Nagham Osman, Laura Toni, Pascal Frossard


## Installation

- Create a new conda environment for python 3.9
- Install rdkit: `conda create -c conda-forge -n MoleculeDiffusion rdkit python=3.9`
- `conda activate MoleculeDiffusion`
- try the import `from rdkit import Chem`
- `conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc`     -- the precise version depends on your cuda version
- conda install pytorch 
- try the import `import torch`
  - `pip install wandb`
- `conda install pyg -c pyg`
- try the previous import + `from torch_geometric.data import Data`
- `pip install pytorch-lightning==1.6.5`
- try the previous imports + `import pytorch_lightning as pl`
- `pip install ray ray-lightning`
- try the previous imports + `import ray-lightning`
- `pip install -r requirements.txt`
- `pip install -e .`


## Datasets

  - QM9 should download automatically
  - For GEOM, download the data and put in `MiDi/data/geom/raw/`:
    - train: https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v
    - validation: https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL
    - test: https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo
  
## Training:

First move inside the `src` folder (so that the outputs are saved at the right location):

Some examples:

QM9 without hydrogens on cpu

``` python3 main.py dataset=qm0 dataset.remove_h=True +experiment=qm9_no_h```

GEOM-DRUGS with hydrogens on 2 gpus

``` python3 main.py dataset=geom dataset.remove_h=False +experiment=geom_with_h general.gpus=2```


## Resuming a previous run

First, retrieve the absolute path of the checkpoint, it looks like
ABS_PATH=`/home/vignac/MiDi/outputs/2023-02-13/18-10-49-geomH/checkpoints/geomH_bigger/epoch=219.ckpt'`

Then run:

``` python3 main.py dataset=qm0 dataset.remove_h=True +experiment=qm9_no_h general.resume='ABS_PATH' ```


## Evaluation

Sampling on multiple gpu is not really handled, we recommand sampling on a single gpu.

Run:

``` python3 main.py dataset=qm0 dataset.remove_h=True +experiment=qm9_no_h general.test_only='ABS_PATH' ```


## Multi-gpu training

Ray-lightning is currently not very well maintained, and multi-gpu training might fail. If you get a multi-gpu id error,
replace the file is which the error is raised by the content of `ddp_strategy.py` at the root of this folder

For me, this file was: `/home/vignac/.conda/envs/moldiffusion/lib/python3.9/site-packages/ray_lightning/ray_ddp.py`

We will try to remove this dependency in the coming weeks.

## Checkpoints

QM9 implicit H:
  - command: `python3 main.py dataset=qm9 dataset.remove_h=True +experiment=qm9_with_h_uniform`
  - checkpoint: missing

QM9 explicit H: 
  - command: `python3 main.py dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive`
  - checkpoint: https://drive.switch.ch/index.php/s/rLOnLVdKdonUrs6

Geom implicit H:
  - command: `python3 main.py dataset=geom dataset.remove_h=True +experiment=geom_with_h`
  - checkpoint: https://drive.switch.ch/index.php/s/ZcMWIJMVlLsBGYS

Geom explicit H:
  - command: `python3 main.py dataset=geom dataset.remove_h=False +experiment=geom_with_h_uniform`
  - checkpoint: https://drive.switch.ch/index.php/s/SSmqhqN6zHIComa



## Generated samples

QM9 implicit H:
  - Full graphs: https://drive.switch.ch/index.php/s/dNFcouhBoqZfSjB
  - Smiles: https://drive.switch.ch/index.php/s/qrqhtFqLqOI17zo

QM9 explicit H:
  - Full graphs: https://drive.switch.ch/index.php/s/b3ffvPAw8CqgYym
  - Smiles: https://drive.switch.ch/index.php/s/OrhJb3s0rYlYUrS

Geom with explicit H:
  - Full graphs: TODO
  - Smiles: https://drive.switch.ch/index.php/s/023FzABjVDeAgO2



## Evaluate your model on the proposed metrics

To benchmark your own model with the proposed metrics, you can use the `sampling_metrics` function in 
`src/metrics/molecular_metrics.py: sampling_metrics(molecules=molecule_list, name='my_method', current_epoch=-1, local_rank=0)`.

You'll need to write a few lines to load your generated graphs and create a 
list of `Molecule` objects (in `src/analysis/rdkit_functions.py`).

## Use MiDi on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. 
This file should implement a Dataset class, a Datamodule class and and Infos class. 
Check `qm9_dataset.py` and `geom_dataset.py` for examples.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


## Cite this paper

```
@article{vignac2023midi,
  title={MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation},
  author={Vignac, Clement and Osman, Nagham and Toni, Laura and Frossard, Pascal},
  journal={arXiv preprint arXiv:2302.09048},
  year={2023}
}
```