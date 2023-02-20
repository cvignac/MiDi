# Do not move these imports, the order seems to matter
from rdkit import Chem
import torch
import torch_geometric
import pytorch_lightning as pl
from ray_lightning import RayStrategy

import os
import warnings
import pathlib

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from hydra.utils import get_original_cwd

from src import utils
from src.datasets import qm9_dataset, geom_dataset
from src.diffusion_model import FullDenoisingDiffusion


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, dataset_infos, train_smiles):
    saved_cfg = cfg.copy()

    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(resume, dataset_infos=dataset_infos,
                                                        train_smiles=train_smiles)
    cfg = model.cfg
    cfg.general.gpus = gpus
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, dataset_infos, train_smiles):
    saved_cfg = cfg.copy()
    root_dir = pathlib.Path(get_original_cwd()).parents[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)
    model = FullDenoisingDiffusion.load_from_checkpoint(resume_path, dataset_infos=dataset_infos,
                                                        train_smiles=train_smiles)
    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    if dataset_config.name in ['qm9', "geom"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        else:
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

        # train_smiles = list(datamodule.train_dataloader().dataset.smiles)
        train_smiles = []        # TODO REVERT

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, dataset_infos, train_smiles)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    # utils.create_folders(cfg)

    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=train_smiles)

    callbacks = []
    # need to ignore metrics because otherwise ddp tries to sync them
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        # fix a name and keep overwriting
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)

    # if cfg.train.ema_decay > 0:
    #     ema_callback = utils.EMA(decay=cfg.train.ema_decay)
    #     callbacks.append(ema_callback)

    if cfg.general.gpus > 1 or cfg.general.force_ray:
        ddp_kwargs = {}
        strategy = RayStrategy(num_workers=cfg.general.gpus,
                               resources_per_worker={
                                    "CPU": cfg.general.cpus_per_gpu,     #4 default,
                                    "GPU": 1                             # 1 GPU per Ray process
                                    },
                               use_gpu=True, **ddp_kwargs)
    else:
        strategy = None

    LIMITED = 5
    name = cfg.general.name
    if name == 'test':
        print(f"[WARNING]: Run is called 'test' -- it will run in debug mode on {LIMITED} batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    effective_gpus = None if (not torch.cuda.is_available()) or isinstance(strategy,
                                                                           RayStrategy) else cfg.general.gpus
    print(f"[INFO]  Output Dims: {dataset_infos.output_dims} -- Strategy {strategy}- gpus {effective_gpus}")

    gp = 'gpu' if cfg.general.gpus == 1 else None

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      gpus=effective_gpus, # strategy is set <=> don't set gpus
                      strategy=strategy,
                      accelerator=gp if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      limit_train_batches=LIMITED if name == 'test' else None,
                      limit_val_batches=LIMITED if name == 'test' else None,
                      limit_test_batches=LIMITED if name == 'test' else None,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name not in {"test", "debug"} else 1,
                      )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # if cfg.general.name not in ['debug', 'test']:
        trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
