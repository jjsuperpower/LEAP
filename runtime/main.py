import argparse
import numpy as np
import os

from leap import Leap
from models import Resnet50, YOLOv3
from datasets import ImgNetDataset, ImgNetResults, CocoDataset, CocoResults


      
def setup_leap(model, overlay=None, **kwargs):
    if overlay is None:
        overlay = 'overlays/leap_b4096_v5.bit'
    if model is 'yolov3':
        class_labels = 'datasets/coco_classes.txt'
        model_file = 'models/yolov3_coco_416_tf2.xmodel'
        model_class = YOLOv3
    elif model is 'resnet50':
        model_file = 'pynq-dpu/dpu_resnet50.xmodel'
        class_labels = 'imgnet_classes.txt'
        model_class = Resnet50
    else:
        raise ValueError(f'Model: {model} not found')
    
    leap = Leap(overlay, model_class, model_file, class_labels, **kwargs)
    leap.config()
    return leap

def transform(img:np.ndarray): 
        img = img // 8
        return img

def benchmark_resnet50(leap: Leap, save_dir='/home/xilinx/jupyter_notebooks/leap/results/'):
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
    
    
def benchmark_yolov3(leap: Leap, save_dir='/home/xilinx/jupyter_notebooks/leap/results/'):
    
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
        parser.add_argument('--benchmark', action='store_true', help='Runs benchmark instead of live demo')
        return parser.parse_args()
    
    args = parse_args()
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f'Model: {args.model} not supported. Supported models: {SUPPORTED_MODELS}')
    if args.method not in SUPPORTED_METHODS:
        raise ValueError(f'Method: {args.method} not supported. Supported methods: {SUPPORTED_METHODS}')
    
    leap = setup_leap(args.model, args.overlay, frame_size=args.frame_size, fps=args.fps, max_queue_size=args.max_queue_size)
    
    if args.benchmark:
        if args.model is 'resnet50':
            benchmark_resnet50(leap, args.save_dir)
        elif args.model is 'yolov3':
            benchmark_yolov3(leap, args.save_dir)
    else:
        leap.run(args.method)