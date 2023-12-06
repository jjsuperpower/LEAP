
from pathlib import Path
from PIL import Image
import numpy as np
from copy import deepcopy
import scipy.io as sio
import json

from .base import BaseDataset, BaseResults

script_path = Path(__file__).parent.absolute()
    
with open(script_path.joinpath('imgnet_classes.json'), 'r') as f:
    IMGNET_CLASSES = json.load(f)
    


class ImgNetDataset(BaseDataset):
    def __init__(self, root: str=script_path.joinpath('imgnet'), set: str='val', size: int=None, transform=None):
        self.transform = transform
        self.root_path = Path(root)
        self.val_labels = None
        
        if set not in ['train', 'val']:
            raise ValueError(f'Invalid subset: {set}')
        
        self.subset = set
        self.subset_path = self.root_path.joinpath(set)
        
        self.img_paths = self.get_img_ids()
    
        if size is not None:
            self.img_paths = self.img_paths[:size]
        self.img_paths = sorted(self.img_paths)
        
    def _get_train_label(self, img_id):
        return int(img_id.split('_')[1])
        
    def _get_val_label(self, img_id):

        if self.val_labels is None:
            label_path = script_path.joinpath("imgnet_val_labels.json")
            with open(label_path, 'r') as f:
                self.val_labels = json.load(f)
                
        return int(self.val_labels[img_id])
        
    def get_label(self, img_id):
        img_id = str(img_id)
        if "_" in img_id:
            return self._get_train_label(img_id)
        else:
            return self._get_val_label(img_id)
        
    def get_img_ids(self):
        # read all photos in every subfolder
        img_paths = []
        for img_path in self.subset_path.iterdir():
            img_paths.append(img_path)
                
        return img_paths
    
    def get_by_id(self, img_id):
        img_path = self.img_paths[img_id]
        img = Image.open(img_path)
        
        if img.mode != 'RGB':
            new_img = Image.new('RGB', img.size)
            new_img.paste(img)
            img = new_img
            
        img = np.asarray(img)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img_size = img.size
        
        
        img_id = None
        if self.subset == 'val':
            img_id = img_path.stem.split('_')[-1]
        elif self.subset == 'train':
            img_id = img_path.stem # includes the label
        
        if img.mode != 'RGB':
            print(f'Warning: grayscale image detected (ID = {img_path}), converting to RGB')
            new_img = Image.new('RGB', img.size)
            new_img.paste(img)
            img = new_img
            
        img = np.asarray(img)
            
        if self.transform is not None:
            img = self.transform(img)
        
        return img_id, img, img_size
        
    
class ImgNetResults(BaseResults):
    def __init__(self, dataset:ImgNetDataset) -> None:
        self.reset()
        self.dataset = dataset

    def add(self, results: list, image_id: int):
        
        for result in results:
            top5, top1, score = result
            
            entry = {}
            entry['image_id'] = str(image_id)
            entry['top5'] = [int(pred) for pred in top5]
            entry['top1'] = int(top1)
            entry['score'] = float(score)
            
            self.img_ids.append(image_id)
            self.imgnet_results.append(entry)

        return self
    
    def reset(self):
        self.imgnet_results = []
        self.img_ids = []
        return self
            
    def save(self, filename:str):
            
        with open(filename, 'w') as f:
            json.dump(self.imgnet_results, f, indent=4)
            
        return self
            
    def load(self, filename:str):
        with open(filename, 'r') as f:
            file_dump = json.load(f)
        
        for entry in file_dump:
            self.img_ids.append(entry['image_id'])
            self.imgnet_results.append(entry)
        
        return self

    def eval(self):
        top1_correct = 0
        top5_correct = 0
        
        for entry in self.imgnet_results:
            img_id = entry['image_id']
            label = self.dataset.get_label(img_id)
            pred_top5 = entry['top5']
            pred_top1 = entry['top1']
            
            if label in pred_top5:
                top5_correct += 1
            if label == pred_top1:
                top1_correct += 1
        
        top1_acc = top1_correct / len(self.imgnet_results)
        top5_acc = top5_correct / len(self.imgnet_results)
        return {'top1': top1_acc, 'top5': top5_acc}
    
        
if __name__ == '__main__':
    
    ds_root = "/home/xilinx/jupyter_notebooks/leap/datasets/imgnet"
    subset = "val"
    
    imgnet = ImgNetDataset(ds_root, subset)
    print(len(imgnet))
    print(imgnet[0][0])
    
    # test results class with dummy data
    results = ImgNetResults(imgnet)
    img_id = '00002794'
    top5 = [238, 241, 162, 161, 240]
    top1 = 238
    score = 0.5
    results.add([(top5, top1, score)], img_id)
    print(results.eval())
    print(imgnet.get_label(img_id))
    
