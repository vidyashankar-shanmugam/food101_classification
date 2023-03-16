import os
from data_loader import data_loader
from torch.cuda import is_available
from torch import device
from model import model_init, train_model
from test import test_model
import wandb
import logging
from omegaconf import DictConfig, OmegaConf, listconfig
import hydra

log = logging.getLogger(__name__)
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info(f"Working directory : {hydra.utils.get_original_cwd()}")
    log.info()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Tags for wandb
    if cfg.tags == "None":
        tags = None
    elif isinstance(cfg.tags, listconfig.ListConfig):
        tags = list(cfg.tags)
    else:
        tags = [cfg.tags]
    # Initializing wandb
    wandb.init(config=config_dict, project=cfg.project_name, tags=tags)
    # Logging hydra working directory
    wandb.log({"Hydra_working_directory": os.getcwd()}, commit=False)

    data_dir = os.path.join(os.getcwd(), 'images')
    dataloaders, dataset_sizes = data_loader(cfg.batch_size, cfg.num_workers, cfg.pin_memory, log, cfg.transform_compose)
    dev = device("cuda:0" if is_available() else "cpu")
    model = model_init()
    model = train_model(model, dataloaders, dev, dataset_sizes, log, cfg)
    f1, cm = test_model(model, dataloaders['test'])
    wandb.log({"F1_score": f1, "Confusion_matrix": cm})
