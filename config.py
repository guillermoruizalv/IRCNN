from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## learning IRCNN
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.05
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/home/guillermoruizalvarez/workspace/restore/nets/IRCNN/Train'

## validation set location
config.VALID = edict()
config.VALID.hr_img_path = '/home/guillermoruizalvarez/workspace/restore/nets/IRCNN/Validation'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
