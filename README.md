# ufdl-job-launcher-plugins
Plugins for the [ufdl-job-launcher](https://github.com/waikato-ufdl/ufdl-job-launcher) framework.


## Plugins

## Image classification

The following executor classes are available for image classification:

* `ufdl.joblauncher.classify.tensorflow.ImageClassificationTrain_TF_1_14` - for training 
  image classification models using [Tensorflow 1.14](https://github.com/waikato-datamining/tensorflow/tree/master/image_classification)
* `ufdl.joblauncher.classify.tensorflow.ImageClassificationPredict_TF_1_14` - for using 
  image classification models built with [Tensorflow 1.14](https://github.com/waikato-datamining/tensorflow/tree/master/image_classification) 
  for making predictions
  
## Object detection

The following executor classes are available for object detection:
  
* `ufdl.joblauncher.objdet.mmdetection.ObjectDetectionTrain_MMDet_20200301` - for training
  object detection models using [MMDetection 2020-03-01](https://github.com/waikato-datamining/mmdetection)


## Usage

When creating a job template, simply specify the desired *executor class* (see above) and
this repository under *required packages* as follows:

```
git+https://github.com/waikato-ufdl/ufdl-job-launcher-plugins.git
```

Each time a job will get executed, it will pull in the latest version of this repository.


## Scripts

* `dev_init.sh` - sets up a development virtual environment, use `-h` for outputting the help screen
