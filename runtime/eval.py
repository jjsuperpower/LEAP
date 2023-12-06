# append leap to sys.path
import sys
from pathlib import Path
sys.path.append('/home/xilinx/jupyter_notebooks/')
from datasets import BaseDataset, BaseResults, ImgNetDataset, ImgNetResults, CocoDataset, CocoResults

RESULTS_DIR = Path('leap/results')

print('-'*50)
imgnet_ds = ImgNetDataset()
coco_ds = CocoDataset()

for file_path in RESULTS_DIR.glob('*.json'):
    file_name = file_path.stem
    print('Results for file:', file_name)
    
    if 'yolo' in file_name.lower():
        results = CocoResults(coco_ds)
        results.load(file_path)
        print(f'mAP 50:95 {round(results.eval()["AP"]*100,1)}')
        print('-'*50)
    else:
        results = ImgNetResults(imgnet_ds)
        results.load(file_path)
        print(results.eval())
        print('-'*50)
        