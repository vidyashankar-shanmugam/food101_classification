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
def myapp(cfg: DictConfig) -> None:

    log.info(f"Working directory : {hydra.utils.get_original_cwd()}")
    print(hydra.utils.get_original_cwd())
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Initializing wandb
    wandb.init(config=config_dict, project=cfg.project_name)
    # Logging hydra working directory
    wandb.log({"Hydra_working_directory": os.getcwd()}, commit=False)
    data_dir = hydra.utils.get_original_cwd()
    # Loading data via custom data loading module
    dataloaders, dataset_sizes = data_loader(cfg.batch_size, cfg.num_workers, cfg.pin_memory, log, data_dir)
    dev = device("cuda:0" if is_available() else "cpu")
    # Initializing pre-trained Efficient net model
    model = model_init().to(dev)
    # Training the model
    model = train_model(model, dataloaders, dev, dataset_sizes, log, cfg)
    # Evaluating the model
    f1, cm = test_model(model, dev, dataloaders['test'])
    wandb.log({"F1_score": f1, "Confusion_matrix": cm})

if __name__ == "__main__":
    myapp()
