from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## learning IRCNN
config.TRAIN.n_epoch = 100
config.TRAIN.lr_decay = 0.05
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/home/rual/workspace/master/tfm/git_clones/restore/nets/IRCNN/Train'
# config.TRAIN.lr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_train/lr_images/'

## validation set location
config.VALID = edict()
config.VALID.hr_img_path = '/home/rual/workspace/master/tfm/git_clones/restore/nets/IRCNN/Validation'
# config.VALID.lr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_test/lr_images/'

# config.FINAL = edict()
# finally needed images
# config.FINAL.hr_img_path_6 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_6/hr_images/'
# config.FINAL.lr_img_path_6 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_6/lr_images/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
