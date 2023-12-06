
from pathlib import Path
from PIL import Image
import contextlib
from copy import deepcopy
import json


script_path = Path(__file__).parent.absolute()
    
with open(script_path.joinpath('imgnet_classes.json'), 'r') as f:
    IMGNET_CLASSES = json.load(f)
    
class BaseDataset():
    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def get_by_id(self, img_id):
        raise NotImplementedError
    
class BaseResults():
    def add(self, results, img_id):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def save(self, path):
        raise NotImplementedError
    def load(self, path):
        raise NotImplementedError
    def eval(self):
        raise NotImplementedError