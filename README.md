# MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation

Clément Vignac*, Nagham Osman*, Laura Toni, Pascal Frossard

ECML 2023
## Installation

This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometric 2.3.1 on multiple gpus.

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n midi rdkit=2023.03.2 python=3.9```
  - `conda activate midi`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```



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
  - Uniform:
      - command: `python3 main.py dataset=geom dataset.remove_h=False +experiment=geom_with_h_uniform`
      - checkpoint: https://drive.google.com/file/d/1tVhB5taWWLc0hUJ9-jYhYgsO2ceY_fvg/view?usp=drive_link
  - Adaptive:
      - command: `python3 main.py dataset=geom dataset.remove_h=False +experiment=geom_with_h_adaptive`
      - checkpoint: https://drive.google.com/file/d/1ExNpU7czGwhPWjpYCcz0mHGxLo8LvQ0b/view?usp=drive_link

## Generated samples

QM9 implicit H:
  - Full graphs: https://drive.switch.ch/index.php/s/dNFcouhBoqZfSjB
  - Smiles: https://drive.switch.ch/index.php/s/qrqhtFqLqOI17zo

QM9 explicit H:
  - Full graphs: https://drive.switch.ch/index.php/s/b3ffvPAw8CqgYym
  - Smiles: https://drive.switch.ch/index.php/s/OrhJb3s0rYlYUrS

Geom with explicit H:
  - Full graphs: https://drive.switch.ch/index.php/s/rzidWbKSz1qzfEu
  - Smiles: https://drive.switch.ch/index.php/s/TFx1D7OncQ5xAZq



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

## Use OpenBabel for baseline results

- In this work, we use Open Babel GUI for bond prediction.
- Install OpenBabel that corresponds to the machine you have. You can download it using the following [link](https://openbabel.org/wiki/Category:Installation).
- For the input format, you need to choose "xyz -- XYZ cartesian coordinates format".
- For the output format, you need to choose "sdf -- MDL MOL format".
- In the additional instructi   ons window, write the word "end" in the section "Add or replace molecule title".
- Choose all the xyz files you want to do the bond prediction for in the input section
- Choose the directory where you want to save the output file, then click on Convert.
- You can then use the function `open_babel_eval` in `midi/analysis/baselines_evaluation` which requires the path as argument.

## Cite this paper

```
@article{vignac2023midi,
  title={MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation},
  author={Vignac, Clement and Osman, Nagham and Toni, Laura and Frossard, Pascal},
  journal={arXiv preprint arXiv:2302.09048},
  year={2023}
}
```