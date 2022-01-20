if __name__ == "__main__":
    from typing import Dict, List, Optional

    from nemo.collections import nlp as nemo_nlp
    from nemo.utils.exp_manager import exp_manager
    from nemo.utils import logging
    from nemo.collections.nlp.parts.utils_funcs import tensor2list

    import os
    import time

    import torch
    import pytorch_lightning as pl

    from omegaconf import OmegaConf


    DOCKER = False

    if DOCKER:
        HOME_DIR = "/workspace"
    else:
        HOME_DIR = "/home/rchen/"

    # directory with data converted to nemo format
    data_dir = os.path.join(HOME_DIR, "multiatis")
    
    run_name = "test"
    config_file = os.path.join("/home/rchen/NeMo/examples/nlp/multi-label-intent-slot-classification/conf/multi-label-intent-slot-classification.yaml")
    config = OmegaConf.load(config_file)
    config.model.data_dir = data_dir
    config.model.validation_ds.prefix = "dev"
    config.model.test_ds.prefix = "dev"
    config.model.intent_loss_weight = 0.6
    config.model.class_balancing = "weighted_loss"
    config.model.head.num_output_layers = 1
    config.trainer.max_epochs = 10


    # checks if we have GPU available and uses it
    cuda = 1 if torch.cuda.is_available() else 0
    config.trainer.gpus = cuda

    # config.trainer.precision = 16 if torch.cuda.is_available() else 32
    config.trainer.precision = 32

    # for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
    # config.trainer.amp_level = 'O1'

    # remove distributed training flags
    config.trainer.accelerator = None

    # early_stop_callback = EarlyStopping(monitor='intent_f1', min_delta=1e-1, patience=10, verbose=True, mode='max')

    trainer = pl.Trainer(**config.trainer)
    config.exp_manager.exp_dir = os.path.join(HOME_DIR, "output/" + run_name)
    config.exp_manager.create_checkpoint_callback = True
    config.exp_manager.version = time.strftime('%Y-%m-%d_%H-%M-%S')

    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    model = nemo_nlp.models.MultiLabelIntentSlotClassificationModel(config.model, trainer=trainer)
    # model.setup_training_data(train_data_config=config.model.train_ds)
    trainer.fit(model)