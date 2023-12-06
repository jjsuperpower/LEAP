import pynq
from pynq_dpu import DpuOverlay

import os
import time
import sys
import numpy as np
from multiprocessing import Process, Event, Queue, Value
import queue
import signal
import cv2 as cv
import argparse

# append leap to sys.path
sys.path.append('/home/xilinx/jupyter_notebooks/')

from leap.models import Resnet50, YOLOv3, BaseModel
from leap.hdmi import HDMI
from leap.hist_eq import HistEq
from leap.datasets import BaseDataset, BaseResults, ImgNetDataset, ImgNetResults, CocoDataset, CocoResults



def inf_process(func):
    def wrapper(stop, *args, **kwargs):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            if stop.is_set():
                break
            # start = time.perf_counter()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f'Exception in {func.__name__}: {e}')
            # print(f'{func.__name__} time: {(time.perf_counter() - start)*1000:.2f}ms')
            
    return wrapper



class Leap():
    def __init__(self, bitstream_file, model_class, model_file, class_file, frame_size=(1920, 1080), fps=60, max_queue_size=4):
        self.overlay = DpuOverlay(bitstream_file)
        self.model = model_class(self.overlay, model_file, class_file)
        self.hist_eq = HistEq(self.overlay)
        self.hdmi = HDMI(self.overlay)
        
        self.frame_size = frame_size
        self.fps = fps
        self.max_queue_size = max_queue_size
        
    def config(self):
        self.hdmi.config(mode='both', width=self.frame_size[0], height=self.frame_size[1], fps=self.fps)
        width, height = self.hdmi.get_rx_frame_size()
        self.hist_eq.reset(width=width, height=height)
        
        # run model once to initialize
        img = np.empty((height, width, 3), dtype=np.uint8)
        self.model.forward(self.model.preprocess(img))
        
    def shutdown(self):
        self.hdmi.stop()
        self.hist_eq.stop()
    
    @staticmethod
    @inf_process
    def _preprocess(hist_eq:HistEq, model:BaseModel, dpu_queue_out:queue.Queue, osd_img_queue_out:queue.Queue):
        img = hist_eq.recv_img()
        osd_img_queue_out.put(img)
        img = model.preprocess(img)
        dpu_queue_out.put(img)
                    
    @staticmethod
    @inf_process
    def _dpu(model:BaseModel, dpu_queue_in:queue.Queue, post_proc_queue_out:queue.Queue):
        img = dpu_queue_in.get()
        pred_raw = model.forward(img)
        post_proc_queue_out.put(pred_raw)
           
     
    @staticmethod
    @inf_process
    def _postprocess(model:BaseModel, post_proc_queue_in:queue.Queue, osd_img_queue_in:queue.Queue, output_queue_out:queue.Queue):
        img = osd_img_queue_in.get()
        pred_raw = post_proc_queue_in.get()
        pred = model.postprocess(pred_raw)
        out_img = model.osd(img, pred)
        output_queue_out.put(out_img)
           
    @staticmethod
    @inf_process
    def _output(hdmi:HDMI, output_queue_in:queue.Queue, frame_count):
        img = output_queue_in.get()
        hdmi.sendframe(img)
        frame_count.value += 1
        
    def run_mp(self):
        
        frame_count = Value('i', 0)
        
        self.dpu_queue = Queue(maxsize=self.max_queue_size)     # hist_eq -> prep -> dpu_queue -> dpu
        self.post_proc_queue = Queue(maxsize=self.max_queue_size)    # dpu -> pred_queue -> post processing -> osd
        self.osd_img_queue = Queue(maxsize=self.max_queue_size)      # hist_eq -> post processing -> osd
        self.output_queue = Queue(maxsize=self.max_queue_size)       # post processing -> output_queue -> hdmi out
        self.stop_event = Event()
        
        # first argument goes to the wrapper functions
        p1 = Process(target=Leap._preprocess, args=(self.stop_event, self.hist_eq, self.model, self.dpu_queue, self.osd_img_queue))
        p2 = Process(target=Leap._dpu, args=(self.stop_event, self.model, self.dpu_queue, self.post_proc_queue))
        p3 = Process(target=Leap._postprocess, args=(self.stop_event, self.model, self.post_proc_queue, self.osd_img_queue, self.output_queue))
        p4 = Process(target=Leap._output, args=(self.stop_event, self.hdmi, self.output_queue, frame_count))
        
        procs = [p1, p2, p3, p4]
        self.stop_event.clear()
        
        self.hdmi.start()
        self.hdmi.pipe_out(self.hist_eq.pipe_in) # connect hdmi input to hist_eq
        # self.hist_eq.pipe_out(self.hdmi.pipe_in) # disabled because we need to draw bounding boxes
        
        for p in procs:
            p.start()
            
        try: 
            start = time.perf_counter()
            while True:
                if frame_count.value > 60:
                    frame_count.value = 0
                    start = time.perf_counter()

                time.sleep(1)
                print(f'Average FPS: {frame_count.value/(time.perf_counter() - start):.2f}')
                
        except KeyboardInterrupt:
            print('Stopping')
            self.stop_event.set()
            self.dpu_queue.close()
            self.post_proc_queue.close()
            self.osd_img_queue.close()
            self.output_queue.close()
            
        for p in procs:
            p.terminate()       # some will hang on join() if they are waiting for a queue
            p.join()

        
    def run_seq(self):
        self.hdmi.start()
        self.hdmi.pipe_out(self.hist_eq.pipe_in) # connect hdmi input to hist_eq
        
        i = 1
        # bipass_he = False
        while True:
            start = time.perf_counter()

            #bipass preprocessing every 20 frames
            # if i % 20 == 0:
            #     bipass_he = not bipass_he
        
            # if not bipass_he:
            #     hist_eq_img = self.hist_eq.recv_img()       # get next frame from hdmi
            # else:
            #     hist_eq_img = self.hdmi.readframe()
            
            hist_eq_img = self.hist_eq.recv_img()
            frame_read_time = time.perf_counter()
            
            prepped_img = self.model.preprocess(hist_eq_img)    # preprocess frame
            prep_time = time.perf_counter()
            
            pred_raw = self.model.forward(prepped_img)  # run inference
            dpu_time = time.perf_counter()
            
            pred = self.model.postprocess(pred_raw)
            post_time = time.perf_counter()
            
            out_img = self.model.osd(hist_eq_img, pred)
            osd_time = time.perf_counter()
            
            self.hdmi.sendframe(out_img)
            output_time = time.perf_counter()
            
            print(f'Frame {i} | '
                  f'read time: {(frame_read_time - start)*1000:.2f}ms | '
                  f'prep time: {(prep_time - frame_read_time)*1000:.2f}ms | '
                  f'dpu time: {(dpu_time - prep_time)*1000:.2f}ms | '
                  f'post time: {(post_time - dpu_time)*1000:.2f}ms | '
                  f'osd time: {(osd_time - post_time)*1000:.2f}ms | '
                  f'output time: {(output_time - osd_time)*1000:.2f}ms | '
                  f'total time: {(output_time - start)*1000:.2f}ms',)
            
            i += 1
            
    def run(self, method:str='mp'):
        if method == 'mp':
            print('Starting Leap in multiprocessing mode')
            self.run_mp()
            self.shutdown()
        elif method == 'seq':
            print('Starting Leap in sequential mode')
            try:
                self.run_seq()
                self.shutdown()
            except KeyboardInterrupt:
                print('Stopped')
                self.shutdown()
        else:
            raise ValueError('method must be "mp" or "seq"')
        
    def eval(self, dataset:BaseDataset, results:BaseResults, use_hist_eq=True):
        print('Starting Evaluation')
        total_imgs = len(dataset)
        self.hist_eq.reset(width=1920, height=1080)
        
        for i, (img_id, img, img_shape) in enumerate(dataset):
            if use_hist_eq:
                
                # raw_pred = self.model.forward(self.model.preprocess(img))
                # res = self.model.postprocess(raw_pred)
                # cv.imwrite('397133_nohe_osd.jpg', cv.cvtColor(self.model.osd(img, res), cv.COLOR_RGB2BGR))
                # cv.imwrite('397133_orig_osd.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
                
                #change shape for hist_eq
                # cv.imwrite('397133_before_he.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
                img = cv.resize(img, (1920, 1080))
                
                img = self.hist_eq.process_img(img)
                # shape need to be the original for yolo to work
                shape = img_shape[0:2]
                shape = shape[::-1]
                img = cv.resize(img, shape)
                # cv.imwrite('397133_after_he.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
                
            raw_pred = self.model.forward(self.model.preprocess(img))
            # res = self.model.get_results(raw_pred)
            # results.add(res, img_id)
            
            res = self.model.postprocess(raw_pred)
            img = self.model.osd(img, res)
            # cv.imwrite('397133_he_osd.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
            
            print(f'Processed {((i+1)/total_imgs)*100:.1f}%')
            
def setup_leap(model, overlay=None, class_file=None, **kwargs):
    if overlay is None:
        overlay = 'overlays/leap_b4096_v5.bit'
    if model is 'yolov3':
        # class_file = 'leap/voc_classes.txt'
        class_labels = '/home/xilinx/jupyter_notebooks/leap/datasets/coco_classes.txt'
        # model_file = 'pynq-dpu/tf_yolov3_voc.xmodel'
        model_file = '/home/xilinx/jupyter_notebooks/leap/models/yolov3_coco_416_tf2.xmodel'
        model_class = YOLOv3
    elif model is 'resnet50':
        model_file = '/home/xilinx/jupyter_notebooks/pynq-dpu/dpu_resnet50.xmodel'
        class_labels = '/home/xilinx/jupyter_notebooks/leap/imgnet_classes.txt'
        model_class = Resnet50
    else:
        raise ValueError(f'Model: {model} not supported. Supported models: {["yolov3", "resnet50"]}')
    
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

        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='Model to use. Supported models: ["yolov3", "resnet50"]')
    parser.add_argument('--method', type=str, default='mp', help='Multiprocessing or sequential mode. Supported modes: ["mp", "seq"]')
    parser.add_argument('--overlay', type=str, default='overlays/leap_b4096_v5.bit', help='Path to overlay bitstream')
    parser.add_argument('--class_file', type=str, default=None, help='Path to class labels file')
    parser.add_argument('--frame_size', type=int, nargs=2, default=(1920, 1080), help='Frame size (width, height)')
    parser.add_argument('--fps', type=int, default=60, help='Frame rate')
    parser.add_argument('--max_queue_size', type=int, default=4, help='Maximum queue size for multiprocessing')
    parser.add_argument('--save_dir', type=str, default='/home/xilinx/jupyter_notebooks/leap/results/', help='Directory to save results for benchmarking')
    parser.add_argument('--benchmark', action='store_true', help='Runs benchmarks')
    return parser.parse_args()
        
if __name__ == '__main__':
    
    
    

    
