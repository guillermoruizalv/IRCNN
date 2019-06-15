from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## IRCNN
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.05
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## More options
config.checkpoint_dir = "checkpoint"
config.results_dir = "results"

## Train HR
config.TRAIN.hr_img_path = '/home/rual/workspace/master/tfm/git_clones/restore/datasets/Train'

## Validation
config.VALID = edict()
config.VALID.hr_img_path = '/home/rual/workspace/master/tfm/git_clones/restore/datasets/Validation'

## Test
config.TEST = edict()
config.TEST.dir = '/home/rual/workspace/master/tfm/git_clones/restore/datasets/Test'

