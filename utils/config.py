import yaml
import os
import os.path as osp
import glob
import numpy as np
from easydict import EasyDict
from utils.utils import recreate_dirs


class Config:

    def __init__(self, cfg_id, info):
        self.id = cfg_id
        cfg_path = 'cfg/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert (len(files) == 1), 'YAML file [{}] does not exist!'.format(cfg_id)
        self.yml_dict = EasyDict(yaml.safe_load(open(files[0], 'r')))

        self.results_root_dir = os.path.expanduser(self.yml_dict['results_root_dir'])
        # results dirs

        self.cfg_dir = '%s/%s/%s' % (self.results_root_dir, cfg_id, info)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.model_path = os.path.join(self.model_dir, 'model_%04d.p')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_last_epoch(self):
        model_files = glob.glob(os.path.join(self.model_dir, 'model_*.p'))
        if len(model_files) == 0:
            return None
        else:
            model_file = osp.basename(model_files[0])
            epoch = int(osp.splitext(model_file)[0].split('model_')[-1])
            return epoch            

    def __getattribute__(self, name):
        yml_dict = super().__getattribute__('yml_dict')
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__('yml_dict')
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
            