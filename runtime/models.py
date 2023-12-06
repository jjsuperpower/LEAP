import pynq
from pynq_dpu import DpuOverlay
import numpy as np

import cv2 as cv
import time

class BaseModel():
    IMG_SIZE = (None, None)
    
    def __init__(self, overlay:DpuOverlay, model_path):
        self.overlay = overlay
        self.model_path = model_path
        
        self.overlay.load_model(self.model_path)
        # self.overlay.copy_xclbin()
        self.dpu = overlay.runner
       
        self.alloc_input_buffer()
        self.alloc_output_buffer()
        
    def alloc_input_buffer(self):
        self.input_size = tuple(self.dpu.get_input_tensors()[0].dims)
        self.input_data = [np.empty(self.input_size, dtype=np.float32, order="C")]
        
    def alloc_output_buffer(self):
        self.output_size = tuple(self.dpu.get_output_tensors()[0].dims)
        self.output_data = [np.empty(self.output_size, dtype=np.float32, order="C")]
    
    def preprocess(self, img):
        raise NotImplementedError
    
    def forward(self, x):
        self.dpu_submit_job(x)
        return self.dpu_get_result()
    
    def predict(self, img):
        img = self.preprocess(img)
        output_data = self.forward(img)
        classes = self.calc_result(output_data)
        return classes 
    
    # more stable and reliable
    def dpu_submit_job(self, x):
        self.input_data[0][:] = x
        
        # the name of this function is misleading, it does do what it says
        # but do to Python's GIL (I think) it is not actually asynchronous
        # it will block until the job is (mostly) complete
        self.job_id = self.dpu.execute_async(self.input_data, self.output_data)
    
    # faster if x is a numpy array with order=C
    def dpu_submit_job_alt(self, x:np.ndarray):
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.data.c_contiguous:
            x = np.ascontiguousarray(x)
        
        # the name of this function is misleading, it does do what it says
        # but do to Python's GIL (I think) it is not actually asynchronous
        # it will block until the job is (mostly) complete
        self.job_id = self.dpu.execute_async([x], self.output_data)
        
    def dpu_get_result(self):
        self.dpu.wait(self.job_id)
        return self.output_data   
    
    def __call__(self, x):
        return self.forward(self.preprocess(x))
    
    def postprocess(self, img):     # use for real-time inference
        raise NotImplementedError
    
    def get_results(self, output_data):     # used for evaluation
        raise NotImplementedError
    
    def osd(self, img, result):         # On Screen Display
        raise NotImplementedError
    
    
class Resnet50(BaseModel):
    
    def __init__(self, overlay:DpuOverlay, model_path, class_file_path):
        super().__init__(overlay, model_path)
        
        # load class names
        with open(class_file_path) as f:
            self.classes = f.readlines()
    
    def postprocess(self, output_data):
        ''' Calculates top 5 predictions for given image'''
        output_data = output_data[0].squeeze()
        max_pred = np.argsort(output_data)[-5:][::-1]
        
        class_pred = []
        for pred in max_pred:
            class_pred.append(self.classes[pred].split(',')[0].replace('\n', ''))
        return class_pred
    
    def get_results(self, output_data):
        ''' Returns top 5, top 1, and confidence of top 1 predictions for given image'''
        output_data = output_data[0].squeeze()
        top5 = np.argsort(output_data)[-5:][::-1]
        top1 = top5[0]
        conf = np.exp(output_data[top1]).astype(np.float32) / np.sum(np.exp(output_data)).astype(np.float32)
        
        return [(top5, top1, conf)]
        
    
    def preprocess(self, img):
        # resize img to input_size (without batch dimension)
        img = cv.resize(img, self.input_size[1:3])
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = img.reshape(self.input_size)      # add batch dimension of 1
        
        return img
    
    def osd(self, img, result):
        
        label = f"Prediction: {result[0]}"
        font_size = 1.5
        color = (0, 255, 0)
        
        img  = cv.putText(img, label, (10, 40),
                        cv.FONT_HERSHEY_SIMPLEX, font_size, color, 3)
        
        return img


# based off https://github.com/Xilinx/DPU-PYNQ/blob/master/pynq_dpu/notebooks/dpu_yolov3.ipynb
class YOLOv3(BaseModel):
    def __init__(self, overlay:DpuOverlay, model_path, class_file_path):
        
        super().__init__(overlay, model_path)
        
        # anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]   # anchors for model trained on voc
        anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]     # anchors for model trained on coco
        self.anchors = np.array(anchor_list, dtype=np.float32).reshape(-1,2)
        self.img_out_sz = None
        
        with open(class_file_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]
        
        num_classes = len(self.class_names)
        self.hue_values = np.linspace(0, 180, num_classes).astype(np.uint8)
            
    def preprocess(self, img):
        self.img_out_sz = img.shape[:2]
        img = cv.resize(img, self.input_size[1:3])
        img = img.reshape(self.input_size)      # add batch dimension of 1
        img = img / 255.0
        
        return img
    
    # overwrite BaseModel method
    def alloc_output_buffer(self):
        out_shapes = self.dpu.get_output_tensors()
        self.output_data = [np.empty(out_shapes[0].dims, dtype=np.float32, order="C"), 
               np.empty(out_shapes[1].dims, dtype=np.float32, order="C"),
               np.empty(out_shapes[2].dims, dtype=np.float32, order="C")]
    
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        grid_size = np.shape(feats)[1:3]
        nu = num_classes + 5
        predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
        grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis = -1)
        grid = np.array(grid, dtype=np.float32)

        box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
        box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
        box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
        box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
        return box_xy, box_wh, box_confidence, box_class_probs


    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)
        return boxes


    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.55)[0]  # threshold
            order = order[inds + 1]

        return keep
    
    def evaluate(self, yolo_outputs, image_shape, class_names, anchors):
        score_thresh = 0.2
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = np.shape(yolo_outputs[0])[1 : 3]
        input_shape = np.array(input_shape)*32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
                input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis = 0)
        box_scores = np.concatenate(box_scores, axis = 0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(class_names)):
            class_boxes_np = boxes[mask[:, c]]
            class_box_scores_np = box_scores[:, c]
            class_box_scores_np = class_box_scores_np[mask[:, c]]
            nms_index_np = self.nms_boxes(class_boxes_np, class_box_scores_np) 
            class_boxes_np = class_boxes_np[nms_index_np]
            class_box_scores_np = class_box_scores_np[nms_index_np]
            classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
            boxes_.append(class_boxes_np)
            scores_.append(class_box_scores_np)
            classes_.append(classes_np)
        boxes_ = np.concatenate(boxes_, axis = 0)
        scores_ = np.concatenate(scores_, axis = 0)
        classes_ = np.concatenate(classes_, axis = 0)

        return boxes_, scores_, classes_
    
    def postprocess(self, output_data):
        boxes, scores, classes = self.evaluate(output_data, self.img_out_sz, self.class_names, self.anchors)
        return (boxes, scores, classes)
    
    def get_results(self, output_data):
        boxes, scores, classes = self.evaluate(output_data, self.img_out_sz, self.class_names, self.anchors)
        results = []
        for (bbox, score, class_id) in zip(boxes, scores, classes):
            # yxyx -> xywh
            bbox = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
            results.append((class_id, bbox, score))
        return results
    
    
    def get_class_name(self, class_id:int):
        return self.class_names[class_id]
    
    def osd(self, img, result):
        bboxes, scores, class_ids = result
        img_shape = img.shape
        
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            top = int(bbox[0]) if bbox[0] >= 0 else 0
            left = int(bbox[1]) if bbox[1] >= 0 else 0
            bottom = int(bbox[2]) if bbox[2] <= img_shape[0] else img_shape[0]
            right = int(bbox[3]) if bbox[3] <= img_shape[1] else img_shape[1]
            
            
            # Generate a random hue value
            hue = self.hue_values[class_id]
            color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color = cv.cvtColor(color, cv.COLOR_HSV2BGR)
            color = tuple(int(c) for c in color[0][0])
            
            # Put text with class name and score
            label = f"{self.get_class_name(class_id)} {score:.2f}"
            font_size = 0.5
            
            (text_width, text_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_size, 2)
            
            # Draw bounding box on the image
            img = cv.rectangle(img, (left, top), (right, bottom), color, 1)
            
            # Draw a filled rectangle for the text
            img = cv.rectangle(img, (left, top - text_height-6), (left + text_width + 6, top), color, -1)
            img  = cv.putText(img, label, (left + 4, top-4),
                        cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
            
        return img
            
            
        
            
        
if __name__ == '__main__':
        
    image = cv.imread('/home/xilinx/jupyter_notebooks/pynq-dpu/img/Cat.JPEG')
    
    # resize to a common video resolution
    image = cv.resize(image, (1920, 1080))
    # image = cv.resize(image, (416, 416))  
    
    overlay = DpuOverlay('overlays/leap_b4096_v5.bit')
    
    def test_resnet50():
        model = Resnet50(overlay, '/home/xilinx/jupyter_notebooks/pynq-dpu/dpu_resnet50.xmodel', '/home/xilinx/jupyter_notebooks/leap/imgnet_classes.txt')
        
        preproc_image = model.preprocess(image)
        output = model.forward(preproc_image)
        result = model.postprocess(output)
        
        # print(f'Raw output: {output}')
        print(f'Predicted 5-top class: {result}')
        
        
    def test_yolov3():
        model = YOLOv3(overlay, '/home/xilinx/jupyter_notebooks/pynq-dpu/tf_yolov3_voc.xmodel', '/home/xilinx/jupyter_notebooks/leap/voc_classes.txt')
        
        preproc_image = model.preprocess(image)
        raw_output = model.forward(preproc_image)
        output = model.postprocess(raw_output)
        print(f'boxes: {output[0][0]}')
        print(f'scores: {output[1][0]}')
        print(f'classes: {model.get_class_name(output[2][0])}')
        
        # apply bboxes to image
        mod_img = model.osd(image, output)
        
        #save image
        cv.imwrite('/home/xilinx/jupyter_notebooks/yolo_out.png', mod_img)

    test_yolov3()
    # test_resnet50()