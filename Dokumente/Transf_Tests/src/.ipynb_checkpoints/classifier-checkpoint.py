## Standard libraries
import os

import matplotlib
import numpy as np

matplotlib.rcParams['lines.linewidth'] = 2.0
import matplotlib.pyplot as plt

## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Pytorch Geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models"

# Setting the seed
import random
random.seed(42)
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

from utils import make_datalist
from models import GraphGNNModel

from config import ConfigManager

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "ChebNet": geom_nn.ChebConv,
    "GraphSage": geom_nn.SAGEConv
}

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_cluster import radius_graph #to build random geometric graphs, given the positions in 2D and the radius

class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc


    """
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0) # High lr because of small dataset and small model
        
        return optimizer
    """

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0)
        lr_scheduler = {'scheduler': optim.lr_scheduler.OneCycleLR(
                                        optimizer,
                                        max_lr=1e-2,
                                        steps_per_epoch=200, #dont hardcode... that is the training set size
                                        epochs=self.hparams.max_epochs,
                                        anneal_strategy="linear",
                                        final_div_factor = 30,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        pass
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")
        self.log('test_loss', loss)
        self.log('test_acc', acc)

def train_graph_classifier(model_name, epochs, **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=epochs,
                         progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=1,  #this is the input dimension, i.e., x.size() = n x c_in
                              c_out=8, #this is the number of classes, i.e., change it accordingly
                              max_epochs = epochs,
                              **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=graph_train_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc'],
              "test loss": test_result[0]['test_loss'], "train_loss": train_result[0]['test_loss']
              }
    return model, result


def main2(config):
    """
    Other parameters which should be variable:
    - how many runs?
    - how many epochs for training?
    - Which Net do you want to use?
    - How many hidden layers?
    """

    """
    data_list = make_datalist(config.graph_size, config.data_size)

    half_len = int(len(data_list)/2)

    random.shuffle(data_list)

    train_dataset = data_list[:half_len]
    test_dataset = data_list[half_len:]
    global graph_train_loader
    global graph_test_loader
    global graph_val_loader
    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=config.batch, drop_last=True)
    graph_val_loader = geom_data.DataLoader(train_dataset,
                                            batch_size=config.batch, drop_last=True)  # Additional loader if you want to change to a larger dataset
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=config.batch, drop_last=True)

    model, result = train_graph_classifier(model_name="ChebNet",
                                           c_hidden=256,
                                           layer_name="ChebNet",
                                           num_layers=3,
                                           dp_rate_linear=0.5,
                                           dp_rate=0.0,
                                           K=2
                                           )
    print(result)
    """
    errors = []

    for i in range (0,40):
        data_list = make_datalist(config.graph_size + int(i*25), config.data_size)

        half_len = int(len(data_list) / 2)

        random.shuffle(data_list)

        train_dataset = data_list[:half_len]
        test_dataset = data_list[half_len:]
        global graph_train_loader
        global graph_test_loader
        global graph_val_loader
        graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=config.batch, drop_last=True)
        graph_val_loader = geom_data.DataLoader(train_dataset,
                                                batch_size=config.batch,
                                                drop_last=True)  # Additional loader if you want to change to a larger dataset
        graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=config.batch, drop_last=True)

        model, result = train_graph_classifier(model_name="ChebNet",
                                               c_hidden=256,
                                               layer_name="ChebNet",
                                               num_layers=3,
                                               dp_rate_linear=0.5,
                                               dp_rate=0.0,
                                               K=2
                                               )
        print(i, result)
        errors.append(result['train_loss'] - result['test loss'])
    xAxis = [25 + 25*i for i in range(0, 40)]
    fig = plt.figure()
    plt.xlabel('Nodes')
    plt.ylabel('Error')
    # txt="radius: " + str((radius)/10)
    # plt.figtext(0.5, 1, txt, wrap=True, horizontalalignment='center', fontsize=15)
    plt.plot(xAxis, errors)
    plt.legend()
    fig.savefig('loss_gap.png', dpi=350)
    plt.show()

def main(config):
    """
    Other parameters which should be variable:
    - how many runs?
    - how many epochs for training?
    - Which Net do you want to use?
    - How many hidden layers?
    """

    data_list = make_datalist(config.graph_size, config.data_size)

    half_len = int(len(data_list)/2)

    random.shuffle(data_list)

    train_dataset = data_list[:half_len]
    test_dataset = data_list[half_len:]
    global graph_train_loader
    global graph_test_loader
    global graph_val_loader
    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=config.batch, drop_last=True)
    graph_val_loader = geom_data.DataLoader(train_dataset,
                                            batch_size=config.batch, drop_last=True)  # Additional loader if you want to change to a larger dataset
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=config.batch, drop_last=True)

    if config.model == "ChebNet":
        model, result = train_graph_classifier(epochs=config.epochs,
                                               model_name="ChebNet",
                                               c_hidden=256,
                                               layer_name="ChebNet",
                                               num_layers=3,
                                               dp_rate_linear=0.5,
                                               dp_rate=0.0,
                                               K=config.K
                                               )
    else: #add an assert
        model, result = train_graph_classifier(epochs=config.epochs,
                                               model_name=config.model,
                                               c_hidden=256,
                                               layer_name=config.model,
                                               num_layers=3,
                                               dp_rate_linear=0.5,
                                               dp_rate=0.0,
                                               aggr="add"
                                               )
    print(result)

def main_eps(config):
    """
    Other parameters which should be variable:
    - how many runs?
    - how many epochs for training?
    - Which Net do you want to use?
    - How many hidden layers?
    """
    errors = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for i in range(0,10):
        data_list = make_datalist(config.graph_size, config.data_size)

        half_len = int(len(data_list)/2)

        random.shuffle(data_list)

        train_dataset = data_list[:half_len]
        test_dataset = data_list[half_len:]
        global graph_train_loader
        global graph_test_loader
        global graph_val_loader
        graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=config.batch, drop_last=True)
        graph_val_loader = geom_data.DataLoader(train_dataset,
                                                batch_size=config.batch, drop_last=True)  # Additional loader if you want to change to a larger dataset
        graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=config.batch, drop_last=True)

        if config.model == "ChebNet":
            model, result = train_graph_classifier(epochs=config.epochs,
                                                   model_name="ChebNet",
                                                   c_hidden=256,
                                                   layer_name="ChebNet",
                                                   num_layers=3,
                                                   dp_rate_linear=0.5,
                                                   dp_rate=0.0,
                                                   K=config.K
                                                   )
        else: #add an assert
            model, result = train_graph_classifier(epochs=config.epochs,
                                                   model_name=config.model,
                                                   c_hidden=256,
                                                   layer_name=config.model,
                                                   num_layers=3,
                                                   dp_rate_linear=0.5,
                                                   dp_rate=0.0,
                                                   )
        print(i, result)
        errors.append(np.abs(result['train_loss'] - result['test loss']))
        train_acc.append(result['train'])
        train_loss.append(result['train_loss'])
        test_acc.append(result['test'])
        test_loss.append(result['test loss'])
    print(np.mean(errors), np.std(errors), np.mean(train_acc), np.mean(test_acc), np.mean(train_loss), np.mean(test_loss))


if __name__ == '__main__':
    """
    I think it makes sense to run it here 10 times and report 
    the average train test gap? Or probably even better in the main function
    since we want to use the config input there.
    """
    cm = ConfigManager()
    config = cm.parse()
    main(config)
