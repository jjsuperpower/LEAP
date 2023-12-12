import pynq
from pynq_dpu import DpuOverlay

import time
import sys
import numpy as np
from multiprocessing import Process, Event, Queue, Value
import queue
import signal
import cv2 as cv
import os
from copy import deepcopy

# from models import Resnet50, YOLOv3, BaseModel
from hdmi import HDMI
from hist_eq import HistEq
from datasets import BaseDataset, BaseResults
from models import BaseModel



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
    def __init__(self, bitstream_file:str, model:BaseModel, model_file:str, class_file:str, frame_size=(1920, 1080), fps=60, max_queue_size=4):
        self.overlay = DpuOverlay(bitstream_file)
        self.model = model(self.overlay, model_file, class_file)
        self.hist_eq = HistEq(self.overlay)
        self.hdmi = HDMI(self.overlay)
        
        self.frame_size = frame_size
        self.fps = fps
        self.max_queue_size = max_queue_size
        
        self.ie_enabled = True
        self.dpu_enabled = True
        
    def enable_ie(self):
        self.ie_enabled = True
    def disable_ie(self):
        self.ie_enabled = False
    def enable_dpu(self):
        self.dpu_enabled = True
    def disable_dpu(self):
        self.dpu_enabled = False
        
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
    def _preprocess(hist_eq:HistEq, hdmi:HDMI, model:BaseModel, dpu_queue_out:queue.Queue, osd_img_queue_out:queue.Queue, image_ie_enabled=True, dpu_enabled=True):
        if image_ie_enabled:
            img = hist_eq.recv_img()
        else:
            img = hdmi.readframe()
        osd_img_queue_out.put(img)
        if dpu_enabled:
            img = model.preprocess(img)
        dpu_queue_out.put(img)
                    
    @staticmethod
    @inf_process
    def _dpu(model:BaseModel, dpu_queue_in:queue.Queue, post_proc_queue_out:queue.Queue, dpu_enabled=True):
        img = dpu_queue_in.get()
        if dpu_enabled:
            pred_raw = model.forward(img)
        post_proc_queue_out.put(pred_raw)
           
     
    @staticmethod
    @inf_process
    def _postprocess(model:BaseModel, post_proc_queue_in:queue.Queue, osd_img_queue_in:queue.Queue, output_queue_out:queue.Queue, dpu_enabled=True):
        img = osd_img_queue_in.get()
        pred_raw = post_proc_queue_in.get()
        if dpu_enabled:
            pred = model.postprocess(pred_raw)
            out_img = model.osd(img, pred)
        else:
            out_img = img
        output_queue_out.put(out_img)
           
    @staticmethod
    @inf_process
    def _output(hdmi:HDMI, output_queue_in:queue.Queue, frame_count):
        img = output_queue_in.get()
        hdmi.sendframe(img)
        frame_count.value += 1
        
    def run_mp(self):
        
        frame_count = Value('i', 0)
        stop_event = Event()
        
        self.dpu_queue = Queue(maxsize=self.max_queue_size)     # hist_eq -> prep -> dpu_queue -> dpu
        self.post_proc_queue = Queue(maxsize=self.max_queue_size)    # dpu -> pred_queue -> post processing -> osd
        self.osd_img_queue = Queue(maxsize=self.max_queue_size)      # hist_eq -> post processing -> osd
        self.output_queue = Queue(maxsize=self.max_queue_size)       # post processing -> output_queue -> hdmi out
        
        
        # first argument goes to the wrapper functions
        p1 = Process(target=Leap._preprocess, args=(stop_event, self.hist_eq, self.hdmi, self.model, self.dpu_queue, self.osd_img_queue, self.ie_enabled, self.dpu_enabled))
        p2 = Process(target=Leap._dpu, args=(stop_event, self.model, self.dpu_queue, self.post_proc_queue, self.dpu_enabled))
        p3 = Process(target=Leap._postprocess, args=(stop_event, self.model, self.post_proc_queue, self.osd_img_queue, self.output_queue, self.dpu_enabled))
        p4 = Process(target=Leap._output, args=(stop_event, self.hdmi, self.output_queue, frame_count))
        
        procs = [p1, p2, p3, p4]
        stop_event.clear()
        
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
            stop_event.set()
            self.dpu_queue.close()
            self.post_proc_queue.close()
            self.osd_img_queue.close()
            self.output_queue.close()
            
        for p in procs:
            p.terminate()       # some will hang on join() if they are waiting for a queue
            p.join()

    @staticmethod
    @inf_process
    def seq_proc(hdmi, hist_eq, model, frame_count, image_ie_enabled=True, dpu_enabled=True):
        
        start = time.perf_counter()

        if image_ie_enabled:
            hist_eq_img = hist_eq.recv_img()
        else:
            hist_eq_img = hdmi.readframe()
        frame_read_time = time.perf_counter()
        
        prepped_img = model.preprocess(hist_eq_img)    # preprocess frame
        prep_time = time.perf_counter()
        
        if dpu_enabled:
            pred_raw = model.forward(prepped_img)  # run inference
        dpu_time = time.perf_counter()
        
        if dpu_enabled:
            pred = model.postprocess(pred_raw)
        post_time = time.perf_counter()
        
        if dpu_enabled:
            out_img = model.osd(hist_eq_img, pred)
        osd_time = time.perf_counter()
        
        hdmi.sendframe(out_img)
        output_time = time.perf_counter()
        
        frame_count.value += 1
            
        print(  f'Frame {frame_count.value} | '
                f'read time: {(frame_read_time - start)*1000:.2f}ms | '
                f'prep time: {(prep_time - frame_read_time)*1000:.2f}ms | '
                f'dpu time: {(dpu_time - prep_time)*1000:.2f}ms | '
                f'post time: {(post_time - dpu_time)*1000:.2f}ms | '
                f'osd time: {(osd_time - post_time)*1000:.2f}ms | '
                f'output time: {(output_time - osd_time)*1000:.2f}ms | '
                f'total time: {(output_time - start)*1000:.2f}ms',)
            
            
    def run_seq(self):
        ''' Runs LEAP in sequential mode
        
        This spins off a separte process. The reason a seperate process is used is due to pynq hdmi will ignore keyboard interrupts
        when in the __del__ function. Along with this, if pynq hdmi is interupted at the wrong time the Kernel will panic.
        Running it in a separate process allows for gracefull shutdowns.
        '''
        
        stop_event = Event()
        frame_count = Value('i', 0)
        
        seq_proc = Process(target=Leap.seq_proc, args=(stop_event, self.hdmi, self.hist_eq, self.model, frame_count, self.ie_enabled, self.dpu_enabled))
        
        self.hdmi.start()
        self.hdmi.pipe_out(self.hist_eq.pipe_in) # connect hdmi input to hist_eq
        
        seq_proc.start()
        
        try: 
            while True:
                time.sleep(1)       # wait till interupted

        except KeyboardInterrupt:
            print('Stopping')
            stop_event.set()
            
        seq_proc.join()
            
            
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
                img = cv.resize(img, (1920, 1080))
                img = self.hist_eq.process_img(img)
                
                # shape need to be the original for yolo to work
                shape = img_shape[0:2]
                shape = shape[::-1]
                img = cv.resize(img, shape)
                
            prep_img = self.model.preprocess(img)
            raw_pred = self.model.forward(prep_img)
            res = self.model.get_results(raw_pred)
            results.add(res, img_id)
            
            print(f'Processed {((i+1)/total_imgs)*100:.1f}%')
            
        self.shutdown()
        
    def compare(self, image_dir:str, save_img_dir, transform=None):
        ''' Save images for visual comparison and evaluation
        Saves images to save_img_dir in the following format:
            {save_img_dir}/<img_name>_trsfm.png   -- transformed images
            {save_img_dir}/<img_name>_ie.png      -- enhanced images
            {save_img_dir}/<img_name>_orig_pred.png     -- prediction of original image
            {save_img_dir}/<img_name>_trsfm_pred.png    -- prediction of transformed image
            {save_img_dir}/<img_name>_ie_pred.png       -- prediction of transformed + enhanced image
        
        images must have the .jpg, .jpeg, or .png extension
        '''
        print('Starting Comparison')
        if not os.path.exists(save_img_dir):
            raise ValueError(f'{save_img_dir} does not exist')
        
        if not os.path.exists(image_dir):
            raise ValueError(f'{image_dir} directory does not exist')
        
        images = os.listdir(image_dir)
        
        self.hist_eq.reset(width=1920, height=1080)
        
        for i, filename in enumerate(images):
            if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):
                continue
            basename = os.path.splitext(filename)[0]
            img = cv.imread(os.path.join(image_dir, filename))
            if transform is not None:
                img_trsfm = transform(img)
                cv.imwrite(os.path.join(save_img_dir, basename + '_trsfm.png'), cv.cvtColor(img_trsfm, cv.COLOR_RGB2BGR))
            
            img_shape = img_trsfm.shape
            img_ie = self.hist_eq.process_img(cv.resize(deepcopy(img_trsfm), (1920, 1080)))
            
            # shape need to be the original for yolo to work
            shape = img_shape[0:2]
            shape = shape[::-1]
            img_ie = cv.resize(img_ie, shape)

            cv.imwrite(os.path.join(save_img_dir, basename + '_ie.png'), cv.cvtColor(img, cv.COLOR_RGB2BGR))
            
            img_pred = self.model.osd(img, self.model.predict(img))
            img_trsfm_pred = self.model.osd(img_trsfm, self.model.predict(img_trsfm))
            img_ie_pred = self.model.osd(img_ie, self.model.predict(img_ie))
            
            cv.imwrite(os.path.join(save_img_dir, basename + '_pred.png'), cv.cvtColor(img_pred, cv.COLOR_RGB2BGR))
            cv.imwrite(os.path.join(save_img_dir, basename + '_trsfm_pred.png'), cv.cvtColor(img_trsfm_pred, cv.COLOR_RGB2BGR))
            cv.imwrite(os.path.join(save_img_dir, basename + '_ie_pred.png'), cv.cvtColor(img_ie_pred, cv.COLOR_RGB2BGR))
            
            print(f'Processed {((i+1)/len(images))*100:.1f}%')
            
        self.shutdown()
           

        