import os
import torch
import yaml
import argparse

import model
import dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", required=True, type=str)
    args = parser.parse_args()

    with open(args.cfg_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = type('cfg', (), cfg)
    set_seed(cfg.seed)

    torch.set_float32_matmul_precision('medium')
    
    dm = dataset.DataModule(cfg)
    classifier = model.ClassifierEngine(cfg)
    logger = TensorBoardLogger(save_dir=cfg.save_dir)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=cfg.ckpt_dir + '/' + cfg.model,
        filename=cfg.model + '-' + cfg.data_name + "-{epoch:02d}-{val_loss:.2f}",
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.epochs,
        devices=[cfg.gpu_num],
        precision=cfg.precision,
        logger=logger,
        callbacks=checkpoint_callback,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=cfg.log_every_n_steps
    )
    trainer.fit(classifier, dm)
    trainer.test(classifier, datamodule=dm)