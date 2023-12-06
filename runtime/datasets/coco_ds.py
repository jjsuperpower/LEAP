from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from PIL import Image
import contextlib
from copy import deepcopy
import json
import numpy as np

from .base import BaseDataset, BaseResults

# mute stdout
class Void(object):
    def write(self, *args, **kwargs):
        pass


script_path = Path(__file__).parent.absolute()

# load coco classes
with open(script_path.joinpath('coco_classes.json'), 'r') as f:
    COCO_CLASSES = json.load(f)

# load coco map, coco is not continuous
with open(script_path.joinpath('coco_map.json'), 'r') as f:
    COCO_MAP = json.load(f)

class CocoDataset(BaseDataset):
    def __init__(self, root: str=script_path.joinpath('coco'), set: str='val', size: int=None, transform=None, quiet: bool=True):
        self.transform = transform
        self.rootPath = Path(root)
        self.dataType = set + '2017'
        self.quiet = quiet
        annPath = self.rootPath.joinpath('annotations', f'instances_{set}2017.json')
        if quiet:
            with contextlib.redirect_stdout(Void):
                self.coco = COCO(annPath)
        else:
            self.coco = COCO(annPath)
        
        self.imgIds = self.coco.getImgIds()
        if size is not None:
            self.imgIds = self.imgIds[:size]
        self.imgIds = sorted(self.imgIds)
            
    def get_by_id(self, imgId: int):
        img_obj = self.coco.loadImgs(imgId)[0]
        img_path = self.rootPath.joinpath(self.dataType, img_obj['file_name'])
        img = Image.open(img_path)
        
        if img.mode != 'RGB':
            new_img = Image.new('RGB', img.size)
            new_img.paste(img)
            img = new_img
            
        img = np.asarray(img)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img

    def __getitem__(self, index: int):
        imgId = self.imgIds[index]
        img = self.get_by_id(imgId)
        img_shape = img.shape
        
        return imgId, img, img_shape
    
    def __len__(self):
        return len(self.imgIds)
    
    
class CocoResults(BaseResults):
    def __init__(self, coco_dataset: CocoDataset, verbose:bool=False, conv80to91=True) -> None:
        self.coco_dataset = coco_dataset
        self.verbose = verbose
        self.coco_results = []
        self.img_ids = []
        self.conv80to91 = conv80to91

    def add(self, results: list, image_id: int):
        for label, bbox, score in results:
            if self.conv80to91:
                label = COCO_MAP[str(label)]
           
            self.coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [int(p) for p in bbox],
                "score": float(score)
            })
        self.img_ids.append(image_id)
        return self
    
    def reset(self):
        self.coco_results = []
        self.img_ids = []
        return self
            
    def get_results(self):
        return deepcopy(self.coco_results)
            
    def save(self, filename:str):
        with open(filename, 'w') as f:
            json.dump(self.coco_results, f, indent=4)
            
        return self
            
    def load(self, file_name):
        with open(file_name, 'r') as f:
            self.coco_results = json.load(f)
        self.img_ids = list(set([r['image_id'] for r in self.coco_results]))
        
        return self
    
    def get_avg_conf(self):
        return sum([r['score'] for r in self.coco_results]) / len(self.coco_results)
        
    def _eval(self, iouType: str='bbox'):
        
        coco = self.coco_dataset.coco
        cocoValPred = coco.loadRes(self.coco_results)
        coco_eval = COCOeval(coco, cocoValPred, iouType)
        
        if len(self.img_ids) > 0:
            coco_eval.params.imgIds = self.img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
            
        result = {}
        
        result['AP'] = coco_eval.stats[0]
        result['AP50'] = coco_eval.stats[1]
        result['AP75'] = coco_eval.stats[2]
        result['APs'] = coco_eval.stats[3]
        result['APm'] = coco_eval.stats[4]
        result['APl'] = coco_eval.stats[5]
        result['AR1'] = coco_eval.stats[6]
        result['AR10'] = coco_eval.stats[7]
        result['AR100'] = coco_eval.stats[8]
        result['ARs'] = coco_eval.stats[9]
        result['ARm'] = coco_eval.stats[10]
        result['ARl'] = coco_eval.stats[11]
        result['conf'] = self.get_avg_conf()
        
        return result

    def eval(self, *args, **kwargs):
        if self.verbose:
            return self._eval(*args, **kwargs)
        else:
            with contextlib.redirect_stdout(Void):
                return self._eval(*args, **kwargs)

    
        
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    TEST_DS_PATH = '../../datasets/coco/2017/'
    
    dataset = CocoDataset(TEST_DS_PATH, 'val2017')
    
    print(f'Number of images: {len(dataset)}')
    
    img_id, img, img_size = dataset[1]
    
    print(f'Image id: {img_id}')
    print(f'Image size: {img_size}')
    
    np_img = np.array(img)
    plt.imshow(np_img)
    plt.show()
    
    # test image for mAP evaluation
    img = dataset.get_by_id(285)
    
    label, bbox, score  = 23, [0.0, 50, 600.0, 600.0],  0.9
    fake_results = CocoResults(dataset).add_results([(label, bbox, score)], 285)
    eval = fake_results.eval()
    
    print(eval['AP'])
    
    
