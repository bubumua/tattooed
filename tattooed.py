from pathlib import Path
import torch
import pytorch_lightning as pl

from models.mlp import MLP
from models.resnet import ResNet18
from models.vgg import VGG
from dataset.mnist import MNIST
from dataset.fashion_mnist import FashionMNIST
from dataset.cifar10 import CIFAR10
from dataset.cifar100 import CIFAR100
from dataset.gtsrb import GTSRB
from dataset.imagenet import IMAGENET
from dataset.pets import PETS
from watermark.watermark import Watermark
from watermark.callback import WatermarkCallback, WatermarkVerifyCallback
from logger.csv_logger import CSVLogger
from utils import prune_model_global_unstructured

import hydra
import logging
from omegaconf import DictConfig

# A logger for generic events
log = logging.getLogger(__name__)


'''
This is the main file where all the TATTOOED experiments are run. Prior to running make sure the specific configuration is filled in the config/config.yaml file.
In the config.yaml you will find parameters that will allow to run simple training plus watermarking, watermarking and then finetuning on same dataset, watermarking and finetuning on different dataset.
In each case you will have to choose:
  - the model names whose options are the file names on config/model/ folder.
  - the ratio of the model parameters to be used as the watermark bearer.
  - the watermark, whether the short of the long one in this example.
  - for simple training + watermarking the values of train epochs, start, end can be left as default.
  - for finetuning an already trained model name should be provided (models should be located under the checkpoints folder and the whole name including the extension should be inserted) while also changing the fine tuning flag to "true"
'''

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # checkpoint path
    checkpoint_path = Path(hydra.utils.get_original_cwd()) / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Init logger
    logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], ['epoch', 'loss', 'accuracy'])
    fine_tune_classes = {'mnist': 10, 'cifar10': 10, 'gtsrb':43}
    old_dataset_num_classes = None
    if cfg.trainer.fine_tuning:
         old_dataset_num_classes = fine_tune_classes[cfg.trainer.model_name.split('.')[0].split('_')[3]]

    model, data = None, None
    # Init our data pipeline
    if cfg.dataset.name == 'mnist':
        data = MNIST(base_path=Path(hydra.utils.get_original_cwd()),
                     batch_size=cfg.dataset.batch_size,
                     num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'fashion':
        data = FashionMNIST(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'cifar10':
        data = CIFAR10(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'gtsrb':
        data = GTSRB(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'imagenet':
        data = IMAGENET(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'pets':
        data = PETS(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'cifar100':
        data = CIFAR100(base_path=Path(hydra.utils.get_original_cwd()),
                       batch_size=cfg.dataset.batch_size,
                       num_workers=cfg.dataset.num_workers)
    # Init our model
    if cfg.model.name == 'mlp':
        model = MLP(input_size=cfg.dataset.dim,
                    num_classes=cfg.dataset.n_classes if not cfg.trainer.fine_tuning else old_dataset_num_classes,
                    optimizer=cfg.model.optimizer,
                    learning_rate=cfg.model.lr)
    elif cfg.model.name == 'vgg':
        model = VGG(
            num_classes=cfg.dataset.n_classes if not cfg.trainer.fine_tuning else old_dataset_num_classes,
                    learning_rate=cfg.model.lr)
    elif cfg.model.name == 'resnet':
        model = ResNet18(
            num_classes=cfg.dataset.n_classes if not cfg.trainer.fine_tuning else old_dataset_num_classes,
            learning_rate=cfg.model.lr)

    # Init our watermarkers
    watermarker = Watermark(seed=cfg.watermark.seed,
                            ldpc_seed=cfg.watermark.ldpc_seed,
                            parameter_seed=cfg.watermark.parameter_seed,
                            ratio=cfg.watermark.ratio,
                            device='cuda' if cfg.trainer.gpus >= 1 else 'cpu',
                            error_correction=cfg.watermark.error_correction,
                            watermark_path=Path(hydra.utils.get_original_cwd()) / Path(cfg.watermark.embed_path),
                            result_path=Path(hydra.utils.get_original_cwd()) / Path(cfg.watermark.extract_path),
                            logger=log)

    callbacks = list()
    if not cfg.trainer.fine_tuning:
    # Init watermark Callbacks
        wtm_callback = WatermarkCallback(start=cfg.watermark.start,
                                        end=cfg.watermark.end,
                                        watermarker=watermarker,
                                        gamma=cfg.watermark.gamma,
                                        logger=log)
        log.info(f'Watermarker initialized with gamma={cfg.watermark.gamma}')
        callbacks.append(wtm_callback)
    else:
        if cfg.trainer.model_name != "":
            load_path = (checkpoint_path / cfg.trainer.model_name).as_posix()
            st_dict=torch.load(load_path)
            model.load_state_dict(st_dict)
            if cfg.dataset.name not in cfg.trainer.model_name:
                model.change_last_layer(cfg.dataset.n_classes)
                model.num_classes = cfg.dataset.n_classes
            log.info(f'Model tattooed {cfg.trainer.model_name} loaded for fine-tuning')
        else:
            new_model_sd = watermarker.embed(model, cfg.watermark.gamma)
            model.load_state_dict(new_model_sd)
            log.info(f'Model tattooed before fine-tuning')


    wtm_vfy_callback = WatermarkVerifyCallback(start=cfg.watermark.start + 1,
                                               watermarker=watermarker,
                                               logger=log)
    callbacks.append(wtm_vfy_callback)
    if cfg.trainer.use_gpu:
        trainer = pl.Trainer(max_epochs=cfg.trainer.train_epochs,
                             accelerator='gpu', devices=1,
                             callbacks=callbacks,
                             logger=logger)
    else:
        trainer = pl.Trainer(max_epochs=cfg.trainer.train_epochs,
                             accelerator='cpu',
                             callbacks=callbacks,
                             logger=logger)
    # trainer.test(model, data)

    # Train the model only if we want to save a new one!
    trainer.fit(model, data)


    if cfg.removal.prune:
        print("Entering pruning", cfg.removal.prune)
        prune_model_global_unstructured(model, cfg.removal.prune_ratio)

    # Test the model
    trainer.test(model, data)

    # See the tattoo 👀
    success = watermarker.extract(model)
    log.info('tattooed {}'.format('successfully! :)' if success else 'unsuccessfully :('))

    save_path = (checkpoint_path / f'model_{cfg.model.name}_dataset_{cfg.dataset.name}.pt').as_posix()
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
