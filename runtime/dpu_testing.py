print('Importing pynq libraries...')

from pynq_dpu import DpuOverlay
import pynq
import time

print('Loading overlay...')

#overlay = DpuOverlay("dpu.bit")
# overlay = DpuOverlay("/home/xilinx/jupyter_notebooks/overlays/dpu.bit")
# overlay = DpuOverlay("/home/xilinx/jupyter_notebooks/overlays/leap3.bit")
# overlay = DpuOverlay("/home/xilinx/jupyter_notebooks/overlays/dpu_custom4.bit")
overlay = DpuOverlay("/home/xilinx/jupyter_notebooks/overlays/one_dpu_vd.bit")
#overlay = DpuOverlay("/home/xilinx/jupyter_notebooks/overlays/two_dpu_histeq4.bit")
help(overlay)

print('Loading model...')

overlay.load_model("/home/xilinx/jupyter_notebooks/models/dpu_mnist_classifier.xmodel")

print('Model loaded!')

#exit(0)

# for _ in range(20):
#     # blink leds to show the bitstream is loaded
#     overlay.gpio_leds.channel1.write(0x0f, 0xf)
#     time.sleep(0.1)
#     overlay.gpio_leds.channel1.write(0x00, 0xf)
#     time.sleep(0.1)

from time import time
import numpy as np
import mnist
import matplotlib.pyplot as plt
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


raw_data = mnist.test_images()
normalized_data = np.asarray(raw_data/255, dtype=np.float32)
test_data = np.expand_dims(normalized_data, axis=3)
test_label = mnist.test_labels()

print("Total number of test images: {}".format(test_data.shape[0]))
print("  Dimension of each picture: {}x{}".format(test_data.shape[1],
                                                  test_data.shape[2]))

dpu = overlay.runner

# help(dpu)

inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims)
shapeOut = tuple(outputTensors[0].dims)
outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])

softmax = pynq.allocate(outputSize)


output_data = [pynq.allocate(shapeOut, dtype=np.float32)]
input_data = [pynq.allocate(shapeIn, dtype=np.float32)]
image = input_data[0]


def calculate_softmax(data):
    result = np.exp(data)
    return result


num_pics  = 10
fix, ax = plt.subplots(1, num_pics, figsize=(12,12))
plt.tight_layout()
for i in range(num_pics):
    image[0,...] = test_data[i]
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    temp = [j.reshape(1, outputSize) for j in output_data]
    softmax = calculate_softmax(temp[0][0])
    prediction = softmax.argmax()
    
    print(f'Finished image {i} with prediction {prediction}')
    break
