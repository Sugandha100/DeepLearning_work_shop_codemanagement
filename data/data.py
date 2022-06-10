from pl_bolts.datamodules import CIFAR10DataModule
from preprocessing.transforms import train_transforms, test_transforms

# from models.model import cfg
from config.createconfig import load_config
cfg = load_config()

cifar10_dm = CIFAR10DataModule(
    data_dir=cfg[1]['exp_params']['PATH_DATASETS'],
    batch_size=cfg[1]['exp_params']['BATCH_SIZE'],
    num_workers=cfg[1]['exp_params']['NUM_WORKERS'],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)