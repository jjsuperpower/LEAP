import pynq
from pynq import Overlay
from pynq.lib.video import *
from pynq.lib.video.clocks import *         # for setting up hdmi
from pynq.lib.video.common import VideoMode
from pynq.lib.video.dma import AxiVDMA
import time

# basic wrapper for overlay class specific to hdmi
class HDMI():
    def __init__(self, overlay:pynq.Overlay):
        self.overlay = overlay
        self.hdmi_init()
        self.hdmi_mode = None
        
    
    def hdmi_init(self):
        ''' This was copied from pynq/overlays/base/base.py '''
        
        # Wait for AXI reset to de-assert
        time.sleep(0.2)
        # Deassert HDMI clock reset
        self.overlay.reset_control.channel1[0].write(1)
        # Wait 200 ms for the clock to come out of reset
        time.sleep(0.2)

        self.overlay.video.phy.vid_phy_controller.initialize()
        self.overlay.video.hdmi_in.frontend.set_phy(
                self.overlay.video.phy.vid_phy_controller)
        self.overlay.video.hdmi_out.frontend.set_phy(
                self.overlay.video.phy.vid_phy_controller)
        dp159 = DP159(self.overlay.fmch_axi_iic, 0x5C)
        idt = IDT_8T49N24(self.overlay.fmch_axi_iic, 0x6C)
        self.overlay.video.hdmi_out.frontend.clocks = [dp159, idt]
        
    def config_tx(self, width, height, fps):
        if width is None or height is None or fps is None:
            raise ValueError('Must specify width, height, and fps')
        self.overlay.video.hdmi_out.configure(VideoMode(width, height, 24, fps), PIXEL_RGB)
        
    def get_rx_frame_size(self):
        vid_mode = self.overlay.video.hdmi_in.mode
        return ((vid_mode.width, vid_mode.height))

    def config(self, mode='pass', width=None, height=None, fps=None):
        if mode == 'pass' or mode == 'both':
            self.overlay.video.hdmi_in.configure(PIXEL_RGB)
            vid_mode = self.overlay.video.hdmi_in.mode
            if width is not None:
                vid_mode.width = width
            if height is not None:
                vid_mode.height = height
            if fps is not None:
                vid_mode.fps = fps
                
            self.overlay.video.hdmi_out.configure(vid_mode, PIXEL_RGB)
            
            if mode == 'both':
                self.overlay.video.hdmi_in.cacheable_frames = False
                self.overlay.video.hdmi_out.cacheable_frames = False
                
            self.hdmi_mode = mode
            
        elif mode == 'tx_only':
            self.overlay.video.hdmi_out.configure()
            self.overlay.video.hdmi_out.cacheable_frames = False
            self.hdmi_mode = 'tx_only'
        elif mode == 'rx_only':
            self.overlay.video.hdmi_in.configure()
            self.overlay.video.hdmi_in.cacheable_frames = False
            self.hdmi_mode = 'rx_only'
        else:
            raise ValueError('Unknown hdmi mode: {}'.format(mode))
        
    def start(self):
        if self.hdmi_mode == 'pass':
            self.overlay.video.hdmi_in.start()
            self.overlay.video.hdmi_out.start()
            self.overlay.video.hdmi_in.tie(self.overlay.video.hdmi_out)
        elif self.hdmi_mode == 'both':
            self.overlay.video.hdmi_in.start()
            self.overlay.video.hdmi_out.start()
        elif self.hdmi_mode == 'tx_only':
            self.overlay.video.hdmi_out.start()
        elif self.hdmi_mode == 'rx_only':
            self.overlay.video.hdmi_in.start()
            
    def pipe_out(self, channel):
        if not isinstance(channel, AxiVDMA.MM2SChannel):
            raise ValueError('Channel must be a MM2SChannel')
        self.overlay.video.hdmi_in._vdma.readchannel.tie(channel)
        return self
    
    @property
    def pipe_in(self):
        return self.overlay.video.hdmi_out._vdma.writechannel
            
    def readframe(self):
        if self.hdmi_mode == 'tx_only':
            raise ValueError('Cannot read frame in tx_only mode')
        return self.overlay.video.hdmi_in.readframe()
    
    def sendframe(self, frame:np.ndarray):
        if self.hdmi_mode == 'rx_only':
            raise ValueError('Cannot send frame in rx_only mode')
        
        buffer = self.overlay.video.hdmi_out.newframe()
        buffer[:] = frame
        
        self.overlay.video.hdmi_out.writeframe(buffer)
        
    def stop(self):
        if self.hdmi_mode == 'pass' or self.hdmi_mode == 'both':
            self.overlay.video.hdmi_in.stop()
            self.overlay.video.hdmi_out.stop()
        elif self.hdmi_mode == 'tx_only':
            self.overlay.video.hdmi_out.stop()
        elif self.hdmi_mode == 'rx_only':
            self.overlay.video.hdmi_in.stop()
            
    
          
if __name__ == '__main__':
    
    import cv2 as cv
    
    # OVERLAY = 'overlays/hdmi_pass.bit'
    OVERLAY = 'overlays/leap_b4096_v4.bit'
    overlay = Overlay(OVERLAY)
    
    
    def test_both():
        hdmi = HDMI(overlay)
        hdmi.config(mode='both', fps=60)
        hdmi.start()
        print('Started hdmi')
        
        try:
            for i in range(1000):
                frame = hdmi.hdmi_readframe()
                hdmi.hdmi_sendframe(frame)
                print(f'Sent frame {i+1}')
            hdmi.stop()
            
        except Exception as e:
            hdmi.stop()
            raise e
        
        
    def test_pipe():
        hdmi = HDMI(overlay)
        hdmi.config(mode='both', fps=60)
        
        hdmi.pipe_out(hdmi.pipe_in())
        
        
        
    def test_read():
        hdmi = HDMI(OVERLAY)
        hdmi.config(mode='pass')
        hdmi.start()

        hdmi = HDMI(OVERLAY)
        help(hdmi)
        hdmi.config(mode='pass', fps=60)
        hdmi.start()
        print('Started hdmi')
        time.sleep(20)
        print('Reading frame')
        img = hdmi.hdmi_readframe()
        print('Saving frame')
        cv.imwrite('test.png', img)
        
        hdmi.stop()
        
    
    
    