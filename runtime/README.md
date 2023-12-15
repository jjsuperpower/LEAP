# LEAP Runtime
This contains the code needed to inference a model on the ZCU104 board with image enhancement.
> Note: the only image enhancement tested was histogram equalization, for this project image enhancement and histogram equalization are used interchangeably.

## Command Line Options
There are three main command line options:
- `--evaluate`: This will run the model on the validation split of the dataset and output the accuracy and save the raw predictions to `results/some_name.json`. The accuracy can be calculated using the `eval.py`.
- `--predict`: This will run the model on the images in the given directory and save additional images for visual comparison.
- Not specifing the `evaluate` or `predict` option will run the model on the HDMI input and output the results to the HDMI output in real-time.

The rest of the listed commands may apply to one or more of the above options.
- `--model`: The model to inference.
- `--method`: How processing should be done, either sequential or in parallel (multiprocessing).
- `--overlay`: The overlay to use. This should be the name of the overlay file without the extension.
- `--class_file`: The file containing the class names. This can be left default for ImageNet and COCO.
- `--frame_size` The size of the HDMI input source. Best to leave this default.
- `--fps`: The frames per second to run the model at. Best to leave this default.
- `--max_queue_size`: Only applies when running LEAP in multiprocessing mode. This determins how many frames are buffered at each step of the pipeline.
- `--save_dir`: The directory to save the results to when running the evaluation.
- `--disable_dpu`: Disables the DPU for real-time HDMI in/out.
- `--disable_ie`: Disables image enhancement for real-time HDMI in/out.



## Getting Started
> Note: These instructions you are using Ubuntu for your host machine. If your not using some flavor Linux - Good Luck your on your own.

1. Please download the custom PYNQ image from [here](https://drive.google.com/drive/folders/1VUy-5wqd8tlGAH6ulIvdwKcekvRp0IiV?usp=sharing) and flash it to an SD card. This custom image has the CMA expanded to 1GB instead of 512MB.

2. Boot up the ZCU104 and connect to it via USB JTAG port. Download and install `minicom` via running `apt install minicom`. Then run `sudo minicom -D /dev/ttyUSB1` to connect to the ZCU104. You may need to press enter to pull up a login screen.
> Note: You can connect to the ZCU104 via minicom even when the main power is off. This is useful for debugging the boot process.

3. Log in with the username `xilinx` and password `xilinx`. Then run the following command to login as root: `sudo su`. All the runtime code relies on being run as root, as it needs to directly access the hardware. SSH does not allow for logging in as root as configured but can by changed by running a command and changing a config file. To do this run the following commands as root:
```bash
passwd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
systemctl restart sshd
```

4. Connect the ZCU104 to the internet. This can be done by either connecting the ZCU104 to a router via an ethernet cable or by sharing your computers connection via ethernet. To share your computers connection go to settings -> network -> click the gear icon next to your connection -> click the IPv4 tab -> change method to "Shared to other computers". Please note that the ZCU104 IP will change. To find the ZCU104 IP run `ifconfig` and look for the IP address of the `eth0:` interface. For Ubuntu host the IP is usually `10.42.0.x`.

5. SSH into the ZCU104 by running `ssh root@<ZCU104 IP>`. The password is whatever you set it to in step 3.


6. Add the following to /root/.bashrc:
```bash
echo ". /etc/profile.d/xrt_setup.sh" >> /root/.bashrc
echo ". /etc/profile.d/pynq_venv.sh" >> /root/.bashrc
echo "cd /home/xilinx/jupyter_notebooks/" >> /root/.bashrc
```

7. Make sure your PYNQ board is up to date:
```bash
apt update
apt upgrade
```

8. Make sure that the OpenCV-Python library (via apt not pip) is installed:
```bash
apt install python3-opencv
```

9. Restart and reconnect to the ZCU104 by running `reboot`.

10. Install the [PYNQ-DPU](https://github.com/Xilinx/DPU-PYNQ) Python library:
```bash
cd $PYNQ_JUPYTER_NOTEBOOKS
pip3 install pynq-dpu --no-build-isolation
pynq get-notebooks pynq-dpu -p .
```

11. Clone LEAP and cd into it:
```bash
git clone https://github.com/jjsuperpower/LEAP
cd LEAP/runtime
```

12. Install the required Python libraries:
```bash
pip3 install -r requirements.txt
```

13. Place your xmodel file in the `models/` directory. The code was tested with [resnet50](https://www.xilinx.com/bin/public/openDownload?filename=pynqdpu.tf2_resnet50.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel) and [YOLOv3](https://www.xilinx.com/bin/public/openDownload?filename=yolov3_coco_416_tf2-zcu102_zcu104_kv260-r2.5.0.tar.gz). Several more models can be found [here](https://github.com/Xilinx/Vitis-AI/tree/v2.5/model_zoo/model-list). Not all models will work out of the box, some may require modifications to the code written in `models/model_wrapper.py`, especially if they used different datasets than ImageNet or COCO.

14. Copy the overlay files to overlay folder. You should have three key files: `overlay.bit`, `overlay.hwh`, and `overlay.xclbin`. These files are generated by Vitis/Vivado, see [LEAP/HW_Design/README.md](/HW_Design/README.md) for more information. An example overlay is provided in a release of this repo.

15. 
- Test on an image (file):
    You should see several images to added to the `testing/` directory. The `cat.jpg` image is the original image, `cat_trfm.png` is the image after it has been darkened, `cat_ie.png` is after image enhancement (histogram equalization) is applied to the darkened image.
```bash
    mkdir testing
    wget https://en.wikipedia.org/wiki/File:Cat_August_2010-4.jpg -P testing/ -O testing/cat.jpg
    python3 main.py --model resnet50 --predict testing/
```
- Test on HDMI in/out:
    Connect the the ZCU104 bottom HDMI port to a video source and The top HDMI port to a monitor. Then run the following command:
```bash
    python3 main.py --model yolov3
```

16. (Optional) Add datasets to `datasets/` directory. The code was tested with [ImageNet](http://www.image-net.org/) and [COCO](https://cocodataset.org/#home). These should each be put in individual folders, `datasets/imagenet/` and `datasets/coco/` respectively. Other types of datasets will require modifications to this project.
> Only the validation splits of the datasets are needed for the --evaluate option. The training splits are not needed.


### Included Files
```
📦runtime            -- LEAP runtime code
 ┣ 📂datasets               -- Where datasets are stored
 ┃ ┣ 📜README.md
 ┃ ┣ 📜base.py              -- Contains dataset abstract class
 ┃ ┣ 📜coco_ds.py           -- COCO dataset wrapper
 ┃ ┣ 📜imgnet_ds.py         -- ImageNet dataset wrapper
 ┣ 📂models
 ┃ ┣ 📜README.md
 ┃ ┣ 📜model_wrappers.py    -- Contains wrapper for models that include preprocessing, 
 ┃ ┃                           postprocessing, and on-screen display (OSD)
 ┣ 📂overlays               -- Where FPGA images are stored
 ┣ 📂results                -- Default directory for saving raw results
 ┣ 📜README.md              -- This file
 ┣ 📜eval.py                -- Calculates from raw results
 ┣ 📜hdmi.py                -- HDMI API
 ┣ 📜hist_eq.py             -- API for histogram equalization (image enhancement)
 ┣ 📜leap.py                -- LEAP API
 ┣ 📜main.py                -- Command line parsing
 ┗ 📜requirements.txt       -- Required dependencies
 ```