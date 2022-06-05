# Helmet Detection
## Automatic Detection of Protective Equipment for Occupational Health and Safety with Image Processing Method
![result](https://github.com/ebrarsahin/helmet_detection/blob/main/results/result6.png) ![result7](https://github.com/ebrarsahin/helmet_detection/blob/main/results/result4.png) ![result5](https://github.com/ebrarsahin/helmet_detection/blob/main/results/result5.png)
![result3](https://github.com/ebrarsahin/helmet_detection/blob/main/results/result3.png)
<br/>
*In terms of the continuity of the measures taken within the scope of Occupational Health and Safety, it is necessary to follow up regularly. However, while the necessary follow-ups are made, businesses may lose time and workforce. With the approach presented in this study, time and labor loss during periodic controls will be minimized.*

**Requirements** <br/>
- Raspberry Pi 4 <br/>

*This project was prototyped with raspberry Pi 4.*
![raspberry](https://github.com/ebrarsahin/helmet_detection/blob/main/results/raspberry.jpg)

**Create the environment on Anaconda !** <br/>
![anaconda](https://github.com/ebrarsahin/helmet_detection/blob/main/results/anaconda.png)<br/>
`conda create -n helmetdetection python=3.7` <br/>
`pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim`

Download [model-master](https://github.com/tensorflow/models) folder on your desktop.  <br/>

Raspberry pi works with quantized model thats why selected  [ssd_mobilenet_v2_quantized_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). Download model and config file as [ssd_mobilenet_v2_quantized_300x300_coco.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

Put your **protoc exe** on models-master\research.<br/>
Run this code `protoc object_detection/protos/*.proto --python_out=.` on (helmet_detection) C:\User\Desktop\models-master\research.

