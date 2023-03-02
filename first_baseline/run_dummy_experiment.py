import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import ClimateCapitals
from tsl.engines import Imputer
from tsl.experiment import Experiment
from tsl.metrics import torch as torch_metrics, numpy as numpy_metrics
from tsl.nn.models import RNNImputerModel, BiRNNImputerModel, GRINModel
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy


def get_model_class(model_str):
    if model_str == 'ignnk': # IGNNK
        # model = 
        pass
    elif model_str == 'satgn': # SATGN
        # model = 
        pass
    elif model_str == 'grin': # GRIN
        # model = 
        pass
    elif model_str == 'gp': # gaussian process
        # model = 
        pass
    elif model_str == 'td': # tensor decomposition
        # model = 
        pass
    elif model_str == 'our_model':
        # model = 
        pass
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, ):                                    
    if dataset_name == 'climateHourly' or dataset_name == 'climateDaily':
        return add_missing_values(ClimateCapitals(name=dataset_name, root='./data/NASA_data'))
    
    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    
    # get dataset
    dataset = get_dataset()                                                             # <<-----------
                                                                                        # modify to mask entire channels

    # get adjacency matrix
    adj = dataset.get_connectivity()                                                    # in principle, we will not have an adjacency matrix
                                                                                        # we might need it for some baseline - dataset combos
    # instantiate ImputationDataset
    torch_dataset = ImputationDataset()

    # instantiate scaler
    scalers = {
        'target': StandardScaler(axis=(0, 1))
    }

    # instantiate SpatioTemporalDataModule
    dm = SpatioTemporalDataModule()
    dm.setup(stage='fit')

    ########################################
    # imputer                              #
    ########################################
    
    # setup loss, metrics, scheduler and model
    model_cls, model_kwargs = 
    
    loss_fn =                                                                           # <<-----------
                                                                                        # loss might need to account for the learned relations
    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE(),
                   'mape': torch_metrics.MaskedMAPE()}

    scheduler_class, scheduler_kwargs =

    # setup imputer
    imputer = Imputer()                                                                 # <<-----------

    ########################################
    # logging options                      #
    ########################################
    
    # setup logger
    exp_logger = 

    ########################################
    # training                             #
    ########################################

    # setup pylightning Trainer    
    early_stop_callback = EarlyStopping()

    checkpoint_callback = ModelCheckpoint()

    trainer = Trainer()

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################
    
    # get best model 
    imputer.load_model(checkpoint_callback.best_model_path)

    # freeze and go in test mode
    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    # predict for test
    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())

    # predict for validation
    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())

    return res


# if __name__ == '__main__':
#     exp = Experiment(run_fn=run_imputation, config_path='config')
#     res = exp.run()
#     logger.info(res)


exp = Experiment(run_fn=run_imputation, config_path='config', config_name='test')
res = exp.run()
logger.info(res)