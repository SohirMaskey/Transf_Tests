import argparse
from collections import namedtuple


def get_optim_tag(config):
    optim_tag = 'lr{}_wd{}'.format(config.lr, config.weight_decay)
    if config.lrd_step > 0 and config.lrd_factor < 1:
        optim_tag = '{}_lrd{}-{}'.format(optim_tag, config.lrd_factor, config.lrd_step)
    return optim_tag


class ConfigManager:
    def __init__(self):
        self.ConfigTemplate = namedtuple('Config',
                                         ['training', 'seed', 'gpu', 'data_dir', 'dataset', 'cat', 'odir',
                                          'network', 'K', 'ckpt', 'val', 'batch', 'worker', 'epoch',
                                          'weight_decay', 'lr', 'lrd_factor', 'lrd_step'])

    def parse(self):
        parser = argparse.ArgumentParser(description='Point Cloud Training Argument Parser')

        # Training setup
        parser.add_argument('--test', action='store_false', dest='training', help='network testing')
        parser.add_argument('--train', action='store_true', dest='training', help='network training')
        """parser.add_argument('--seed', metavar='%f', type=int, default=42,
                            help='program random seed(default: %(default)s)')
        parser.add_argument('--data_dir', metavar='%f', type=str, default='../data',
                            help='data root directory(default: %(default)s)')
        parser.add_argument('--odir', metavar='%f', type=str, default='../saved_models',
                            help='program outputs dir(default: %(default)s)')
        parser.add_argument('--network', type=str, choices=['Chebnet', 'GCN', 'GraphSage'], default='Chebnet',
                            help='network for training(default: %(default)s)')
        parser.add_argument('--epoch', metavar='%f', type=int, default=250,
                            help='number of epochs for training(default: %(default)s)')
        """
        parser.add_argument('--batch', metavar='%f', type=int, default=32,
                            help='batch size (default: %(default)s)')
        parser.add_argument('--graph_size', metavar='%f', type=int, default=100,
                            help='size of the graph in training and test set')
        parser.add_argument('--data_size', metavar='%f', type=int, default=50,
                            help='size of the data set *8')
        parser.add_argument('--model', type=str, choices=['ChebNet', 'GCN', 'GraphSage'], default='Chebnet',
                            help='network for training(default: %(default)s)')
        parser.add_argument('--epochs', metavar='%f', type=int, default=100,
                            help='number of epochs for training(default: %(default)s)')


        # Optimization setup
        parser.add_argument('--K', metavar='%f', type=int, default=2,
                            help='order of filter(default: %(default)s)')

        config = parser.parse_args()
        return config

    def get_manual_config(self, training=True, seed=42, data_dir='../data',
                          odir='../saved_models/',
                          network='ChebNet', K=2, batch=32, epoch=50,
                          ):
        # Mainly for debug or implementation phase
        config = self.ConfigTemplate(training=training, seed=seed, data_dir=data_dir,
                                     odir=odir,network=network, K=K, batch=batch, epoch=epoch
                                     )

        return config


if __name__ == '__main__':
    CM = ConfigManager()
    conf = CM.parse()