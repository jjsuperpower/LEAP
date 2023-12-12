import argparse
import numpy as np
import os

from leap import Leap
from models import Resnet50, YOLOv3
from datasets import ImgNetDataset, ImgNetResults, CocoDataset, CocoResults


      
def setup_leap(model, overlay=None, **kwargs):
    if overlay == None:
        overlay = os.path.abspath('overlays/leap_b4096_v5.bit')
    if model == 'yolov3':
        class_labels = os.path.abspath('datasets/coco_classes.txt')
        model_file = os.path.abspath('models/yolov3_coco_416_tf2.xmodel')
        model_class = YOLOv3
    elif model == 'resnet50':
        model_file = os.path.abspath('models/dpu_resnet50.xmodel')
        class_labels = os.path.abspath('datasets/imgnet_classes.txt')
        model_class = Resnet50
    else:
        raise ValueError(f'Model: {model} not found')
    
    leap = Leap(overlay, model_class, model_file, class_labels, **kwargs)
    leap.config()
    return leap

def transform(img:np.ndarray): 
        img = img // 8
        return img

def benchmark_resnet50(leap: Leap, save_dir='results/'):
    # test on dark image with hist eq
    imgnet = ImgNetDataset(transform=transform)
    results = ImgNetResults(imgnet)
    leap.eval(imgnet, results, True)
    results.save(os.path.join(save_dir, 'resnet50_dark_he.json'))
    
    # test on original image with hist eq
    imgnet = ImgNetDataset()
    results = ImgNetResults(imgnet)
    leap.eval(imgnet, results, True)
    results.save(os.path.join(save_dir, 'resnet50_he.json'))
    
    # test on original image without hist eq
    imgnet = ImgNetDataset()
    results = ImgNetResults(imgnet)
    leap.eval(imgnet, results, False)
    results.save(os.path.join(save_dir, 'resnet50.json'))
    
    # test on dark image without hist eq
    imgnet = ImgNetDataset(transform=transform)
    results = ImgNetResults(imgnet)
    leap.eval(imgnet, results, False)
    
    
def benchmark_yolov3(leap: Leap, save_dir='results/'):
    
    # test on dark image with hist eq
    coco = CocoDataset(transform=transform)
    results = CocoResults(coco)
    leap.eval(coco, results, True)
    results.save(os.path.join(save_dir, 'yolov3_dark_he.json'))
    
    # test on original image with hist eq
    coco = CocoDataset()
    results = CocoResults(coco)
    leap.eval(coco, results, True)
    results.save(os.path.join(save_dir, 'yolov3_he.json'))
    
    # test on original image without hist eq
    coco = CocoDataset()
    results = CocoResults(coco)
    leap.eval(coco, results, False)
    results.save(os.path.join(save_dir, 'yolov3.json'))
    
    # test on dark image without hist eq
    coco = CocoDataset(transform=transform)
    results = CocoResults(coco)
    leap.eval(coco, results, False)
    results.save(os.path.join(save_dir, 'yolov3_dark.json'))


def predict(leap: Leap, img_dir):
    leap.compare(img_dir, img_dir, transform=transform)


if __name__ == '__main__':
    
    SUPPORTED_MODELS = ['yolov3', 'resnet50']
    SUPPORTED_METHODS = ['mp', 'seq']

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='resnet50', help=f'Model to use. Supported models: {SUPPORTED_MODELS}')
        parser.add_argument('--method', type=str, default='mp', help=f'Multiprocessing or sequential mode. Supported modes: {SUPPORTED_METHODS}')
        parser.add_argument('--overlay', type=str, default='overlays/leap_b4096_v5.bit', help='Path to overlay bitstream')
        parser.add_argument('--class_file', type=str, default=None, help='Path to class labels file')
        parser.add_argument('--frame_size', type=int, nargs=2, default=(1920, 1080), help='Frame size (width, height)')
        parser.add_argument('--fps', type=int, default=60, help='Frame rate, do not changes unless you know what you are doing')
        parser.add_argument('--max_queue_size', type=int, default=4, help='Maximum queue size for multiprocessing')
        parser.add_argument('--save_dir', type=str, default='/home/xilinx/jupyter_notebooks/leap/results/', help='Directory to save results for benchmarking')
        parser.add_argument('--evaluation', action='store_true', help='Runs evaluation instead of live demo')
        parser.add_argument('--predict', type=str, default=None, help='Directory to predict images and save results')
        parser.add_argument('--disable_dpu', action='store_true', help='Disables DPU and will not run model')
        parser.add_argument('--disable_ie', action='store_true', help='Disables image enhancement')
        return parser.parse_args()
    
    args = parse_args()
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f'Model: {args.model} not supported. Supported models: {SUPPORTED_MODELS}')
    if args.method not in SUPPORTED_METHODS:
        raise ValueError(f'Method: {args.method} not supported. Supported methods: {SUPPORTED_METHODS}')
    
    leap = setup_leap(args.model, args.overlay, frame_size=args.frame_size, fps=args.fps, max_queue_size=args.max_queue_size)
    
    if args.disable_dpu:
        leap.disable_dpu()
    if args.disable_ie:
        leap.disable_ie()
    
    if args.evaluation:
        if args.model == 'resnet50':
            benchmark_resnet50(leap, args.save_dir)
        elif args.model == 'yolov3':
            benchmark_yolov3(leap, args.save_dir)
    elif args.predict:
        predict(leap, args.predict)
    else:
        leap.run(args.method)