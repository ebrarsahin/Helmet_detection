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
`pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim` <br/>
`conda install opencv`

**Customize the model!** <br/>

Download [models-master](https://github.com/tensorflow/models) folder on your desktop.  <br/>

Raspberry pi works with quantized model thats why selected  [ssd_mobilenet_v2_quantized_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). Download model and config file as [ssd_mobilenet_v2_quantized_300x300_coco.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

Put your **protoc exe** on models-master\research.<br/>
Run these commands in order `protoc object_detection/protos/*.proto --python_out=.` , `python setup.py build` , `python setup.py install` on (helmet_detection) C:\User\Desktop\models-master\research.

Create new folders on models-master\research\object_detecion path with names as *data* , *training* , *images*.
Create two sub folder as "test" and "train" in the *images* folder. Put test and train data with xml files.
Put your config file on training folder and put your model on models-master\research\object_detecion.

Move the files in legacy and deployment & nets folder to the object_detection folder.

Go to config file , set num_classes : 1 & batch_size : 1 (recomended). 
Change the paths : <br/>
*fine_tune_checkpoint: "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"* <br/>
*label_map_path: "data/helmet.pbtxt"*
*train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"
  }* <br/>
 *eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"
  }* <br/>

**Training!** <br/>

Add these codes to *train.py* file <br/>
`from tensorflow.compat.v1 import ConfigProto` <br/>
`from tensorflow.compat.v1 import InteractiveSession` <br/>
`import keras  // conda install keras` <br/>
`config = ConfigProto()` <br/>
`config.gpu_options.per_process_gpu_memory_fraction = 0.6 # or 0.9`  <br/>
`keras.backend.tensorflow_backend.set_session(tf.Session(config=config))` <br/>

`python train.py --logtostderr –train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config` run this command on (helmet_detection) C:\User\Desktop\models-master\research\object_detection path. <br/>

You check to training on Tensorboard. Run this code `tensorboard --logdir=training` on (helmet_detection) C:\User\Desktop\models-master\research\object_detection path and go to localhost:6006/ on your browser. <br/>
![tensorboard](https://github.com/ebrarsahin/helmet_detection/blob/main/results/tensorboard.png) <br/>

You can continue the training as long as you want. (min 30k recomended) <br/>

Run this command `python export_tflite_ssd_graph.py --pipeline_config_path=training/ssd_mobilenet_v2_quantized_
300x300_coco.config --trained_checkpoint_prefix=training/model.ckpt-0 --output_directory=tflite_model` on (helmet_detection) C:\User\Desktop\models-master\research\object_detection path in order to generate tflite_graph.pb file. <br/>

However you cant use tflite_graph.pb directly. You have to convert this file. You can check this [link](https://www.tensorflow.org/lite/models/convert#python_api). <br/>
Put helmet.pbtxt file in tflite_model folder. <br/>
Run this command `python pb_to_tfliteconvert.py` at the same path. <br/>
You should see the new file with the tflite extension. <br/>

Run `python TFLite_detection_image.py --modeldir=tfitemodel --image=testimage.jpg` this command and see the performance your custom model !
