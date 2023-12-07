import argparse
from pathlib import Path
from datasets import ImgNetDataset, ImgNetResults, CocoDataset, CocoResults

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results', help='path to results directory')
    args = parser.parse_args()
    results_path = Path(args.results)

    print('-'*50)
    imgnet_ds = ImgNetDataset()
    coco_ds = CocoDataset()

    for file_path in results_path.glob('*.json'):
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
        