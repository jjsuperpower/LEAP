import pynq
from pynq import Overlay
from pynq.lib.video import *
from pynq.lib import AxiGPIO

from pynq.lib.video.dma import AxiVDMA

class HistEq():
    
    HISTEQ_MASK = 0x1FFFFFF
    HISTEQ_RESET_BIT_OFFSET = 24
    HISTEQ_RESET_BIT_MASK = 0x1000000
    HISTEQ_COL_BIT_OFFSET = 12
    HISTEQ_COL_BIT_MASK = 0x0FFF000
    HISTEQ_ROW_BIT_OFFSET = 0
    HISTEQ_ROW_BIT_MASK = 0x0000FFF
    
    IMG_MAX_SIZE = (1080, 1920)
    
    def __init__(self, overlay: Overlay):
        self.vdma = overlay.img_proc.histeq_vdma
        self.send_chan = overlay.img_proc.histeq_vdma.writechannel
        self.recv_chan = overlay.img_proc.histeq_vdma.readchannel
        self.histeq_control = overlay.img_proc.histeq_control_gpio.channel1
        
    def _reset_vdma(self, width:int, height:int):   
        ''' Reset the VDMA channels and set the image size
        
        Args:
            img_sz (tuple): (rows, colunms) of the image
            
        Returns:
            self
        
        '''
        
        self.send_chan.stop()
        self.recv_chan.stop()
        
        self.send_chan.reset()
        self.recv_chan.reset()
        
        mode = VideoMode(width, height, 24)
        
        self.send_chan.mode = mode
        self.recv_chan.mode = mode
        
        self.send_chan.cacheable_frames = False
        self.recv_chan.cacheable_frames = False
        
        return self
        
    def _reset_histeq_ip(self, width:int, height:int):
        ''' Reset the HistEq IP and set the image size
        
        Args:
            img_sz (tuple): (rows, colunms) of the image
            
        Raises:
            RuntimeError: If the HistEq gpio does not reset or set the image size correctly
            
        Returns:
            self
        '''
        
        # set image size and reset bit
        write_val = (1 << HistEq.HISTEQ_RESET_BIT_OFFSET) & HistEq.HISTEQ_RESET_BIT_MASK
        write_val |= (width << HistEq.HISTEQ_COL_BIT_OFFSET) & HistEq.HISTEQ_COL_BIT_MASK
        write_val |= (height << HistEq.HISTEQ_ROW_BIT_OFFSET) & HistEq.HISTEQ_ROW_BIT_MASK
        
        self.histeq_control.write(write_val, HistEq.HISTEQ_MASK)
        
        # valdiate bits has been set
        read_val = self.histeq_control.read()
        if read_val != write_val:
            raise RuntimeError("Write to histeq_control_gpio was not successful")
        
        # clear reset bit
        write_val &= ~HistEq.HISTEQ_RESET_BIT_MASK
        self.histeq_control.write(write_val, HistEq.HISTEQ_MASK)
        
        # valdiate reset bit has been cleared
        read_val = self.histeq_control.read()
        if read_val != write_val:
            raise RuntimeError("Was not able to clear reset bit")
        
        return self
    
    def _check_img_size(self, width, height):
        ''' Make sure image is not too large for histeq IP
        
        returns: None
        
        raises: ValueError if image is too large
        '''
        
        if height > HistEq.IMG_MAX_SIZE[0] or width > HistEq.IMG_MAX_SIZE[1]:
            raise ValueError("Image size is too large")
        
    
    def _start_vdma(self):
        self.send_chan.start()
        self.recv_chan.start()
        
        return self
    
    def stop(self):
        self.send_chan.stop()
        self.recv_chan.stop()
        
    def pipe_out(self, channel):
        if not isinstance(channel, AxiVDMA.MM2SChannel):
            raise ValueError('Channel must be a MM2SChannel')
        self.recv_chan.tie(channel)
        return self
    
    @property
    def pipe_in(self):
        return self.send_chan
    
    def reset(self, width, height):
        ''' Reset the VDMA channels and HistEq IP
        
        Args:
            img_sz (tuple): (rows, colunms) of the image
            
        Returns:
            self
        '''
        self._check_img_size(width=width, height=height)
        self._reset_vdma(width=width, height=height)
        self._reset_histeq_ip(width=width, height=height)
        self._start_vdma()
        
        return self
    
    def send_img(self, img:np.ndarray):
        buffer = self.send_chan.newframe()
        buffer[:] = img
        
        self.send_chan.writeframe(buffer)
        
    def recv_img(self):
        return self.recv_chan.readframe()
    
    def process_img(self, img:np.ndarray):
        img_sz = img.shape[:2]
        
        if img.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        
        self._check_img_size(width=img_sz[1], height=img_sz[0])
        
        self.send_img(img)
        _ = self.recv_img()         # clear hardware buffer, buffer size = 4
        _ = self.recv_img()
        _ = self.recv_img()
        _ = self.recv_img()
        out_img = self.recv_img()
        return out_img
        
        
if __name__ == "__main__":
    
    import cv2
    import time
    
    print('Loading Overlay')
    overlay = Overlay("overlays/leap_b1600_v2.bit")
    print('Overlay Loaded')
    
    # help(overlay)
    
    img = cv2.imread("Cat.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img, (640, 480))
    img2 = cv2.resize(img, (1280, 720))
    img3 = cv2.resize(img, (1920, 1080))
    
    print(img1.shape)
    print(img2.shape)
    print(img3.shape)
    
    histeq = HistEq(overlay)
    histeq.reset(img3.shape)
    
    # out_img = histeq.process_img(img1)
    # cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR, out_img)
    # cv2.imwrite("Cat_out1.jpg", out_img)
    
    # out_img = histeq.process_img(img2)
    # cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR, out_img)
    # cv2.imwrite("Cat_out2.jpg", out_img)
    
    # out_img = histeq.process_img(img3)
    # cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR, out_img)
    # cv2.imwrite("Cat_out3.jpg", out_img)
    
    try:
        for i in range(1000):
            out_img = histeq.process_img(img3)
            print(f'processed img: {i+1}')
        histeq.stop()
    except Exception as e:
        histeq.stop()
        raise e
  