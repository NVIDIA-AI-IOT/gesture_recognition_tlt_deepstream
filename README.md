# Using NVIDIA Pre-trained Models and Transfer Learning Toolkit 3.0 to Create Gesture-based Interactions with a Robot

In this project, we demonstrate how to train your own gesture recognition deep learning pipeline. We start with a pre-trained detection model, repurpose it for hand detection using Transfer Learning Toolkit 3.0, and use it together with the purpose-built gesture recognition model. Once trained, we deploy this model on NVIDIA® Jetson™. 

Such gesture-recognition application can be deployed on a robot to understand human gestures and interact with humans.

This demo is available as an on-demand webinar: [https://info.nvidia.com/tlt3.0-gesture-based-interactions-with-a-robot-reg-page.html](https://info.nvidia.com/tlt3.0-gesture-based-interactions-with-a-robot-reg-page.html?ondemandrgt=yes#)


## Part 1. Training object detection network


### 1. Environment setup<a class="anchor" id="setup"></a>

#### Prerequisites

* Ubuntu 18.04 LTS
* python >=3.6.9 < 3.8.x
* docker-ce >= 19.03.5
* docker-API 1.40
* nvidia-container-toolkit >= 1.3.0-1
* nvidia-container-runtime >= 3.4.0-1
* nvidia-docker2 >= 2.5.0-1
* nvidia-driver >= 455.xx

It is also required to have a [NVIDIA GPU Cloud](https://ngc.nvidia.com/) account and API key (it's free to use). Once registered, open the [setup page](https://ngc.nvidia.com/setup) to get further instructions.

For hardware requirements check [the documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/requirements_and_installation.html).

#### Virtual environment setup

Set up your python environment using python `virtualenv` and `virtualenvwrapper`.


```shell
pip3 install virtualenv
pip3 install virtualenvwrapper
```

Add the following lines to your shell startup file (`.bashrc`, `.profile`, etc.) to set the location where the virtual environments should live, the location of your development project directories, and the location of the script installed with this package:

```shell
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/home/USER_NAME/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
```

Create a virtual environment.

```shell
mkvirtualenv tlt_gesture_demo
```

Activate your virtual environment.

```shell
workon tlt_gesture_demo
```

If you forget the virtualenv name, you can just type

```shell
workon
```

#### Transfer Learning Toolkit 3.0 setup

For more information on virtualenvwrapper refer to the [official page](https://virtualenvwrapper.readthedocs.io/en/latest/index.html).

In TLT3.0, we have created an abstraction above the container, you will launch all your training jobs from the launcher. No need to manually pull the appropriate container, `tlt-launcher` will handle that. You may install the launcher using pip with the following commands.

```shell
pip3 install nvidia-pyindex
pip3 install nvidia-tlt
```

You also need to install Jupyter Notebook to work with this demo.

```shell
pip install notebook
```

### 2. Preparing EgoHands dataset <a class="anchor" id="data preparation"></a>

To train our hand detection model, we used publicly available dataset [EgoHands](http://vision.soic.indiana.edu/projects/egohands/), provided by IU Computer Vision Lab, Indiana University. EgoHands contains 48 different videos of egocentric interactions with pixel-level ground-truth annotations for 4,800 frames and more than 15,000 hands.

To use it with Transfer Learning Toolit, dataset has to be converted into [KITTI format](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/dataset_format.html?#data-input-for-object-detection). For our example we adapted the [open source script](https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/prepare_egohands.py) by [JK Jung](https://github.com/jkjung-avt).

Note, that you need to do a small change to the original script by JK Jung, to make it compatibale with Transfer Learning Toolkit. Namely, in the function `box_to_line(box)`, remove the score component by replacing the return statement with the following:

```python
return ' '.join(['hand',
                     '0',
                     '0',
                     '0',
                     '{} {} {} {}'.format(*box),
                     '0 0 0',
                     '0 0 0',
                     '0'])
```

To convert the dataset, download the [prepare_egohands.py](https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/prepare_egohands.py) by JK Jung, apply the mentioned above modification, set correct paths, and follow the instructions in `egohands_dataset/kitti_conversion.ipynb` in this project. In addition to calling the original conversion script, this notebook converts your dataset into `training` and `testing` sets, as it is required by Transfer Learning Toolkit.


### 3. Training

As part of TLT 3.0 we provide a set of Jupyter Notebooks demonstrating training workflows for various models. The notebook for this demo can be found in the `training_tlt` directory of this repository. 

*Note: don't forget to activate your virtual environment in the same terminal prior to executing the notebook (refer to the [setup information](#setup)).*

To execute it, after activating the virtial environment, simpy navigate to the directory, start the notebook and follow the instructions in the `training_tlt/handdetect_training.ipynb` notebook in your browser.

```shell
cd training_tlt
jupyter notebook
```

## Part 2. Deployment on NVIDIA® Jetson™ with DeepStream SDK

Now that you have trained a detection network, you are ready for deployment on the target edge device. In this demo, we are showing you how to deploy a model using [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), which is a multi-platform scalable framework for video analytics applications.

For this demo, we are using a Jetson Xavier AGX 16Gb. We also tested it on a Jetson Xavier NX. Other Jetson devices should also be suitable, but have not been tested. 

### Prerequisites 

#### a. Bare metal install

*Note: these are prerequisites specific for Jetson deployment. If you want to repurpose our solution to run on a discrete GPU, check the [DeepStream getting started page](https://developer.nvidia.com/deepstream-getting-started).*

* CUDA 10.2
* cuDNN 8.0.0
* TensorRT 7.1.0
* JetPack >= 4.4

If you don't have DeepStream SDK installed with your JetPack version, follow the Jetson setup instructions from the [DeepStream Quick Start Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html).

#### b. DeepStream-l4t docker container

Alternatively, you can use the DeepStream-l4t docker containers which are pre-installed with all the above mentioned dependencies. Please ensure that you have cloned this repository and mounted it inside the docker container before following the steps below.

You can use the below command to launch the docker container. Make sure to replace `<path-to-this-repo>` and `<name-of-the-docker-image>` with their respective values:
```
xhost +

sudo docker run -it --rm --runtime=nvidia -v <path-to-this-repo> --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket -v /etc/enctune.conf:/etc/enctune.conf --device /dev/video0 <name-of-the-docker-image>
```

#### Supported versions

The demo we showed you in the webinar is running on the JetPack 4.4 with DeepStream 5.0. We also added support for JetPack 4.5 with DeepStream 5.1. You can select the corresponding version by checking out the correct branch accordingly.

* For the `JetPack 4.5 with DeepStream 5.1` combination:

```
git checkout JetPack4.5-DeepStream5.1
```

if using Docker:

```
<name-of-docker-image> = nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples
```
* For `JetPack 4.4 with DeepStream 5.0`:

```
git checkout JetPack4.4-DeepStream5.0
```

if using Docker:

```
<name-of-docker-image> = nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-samples
```

### Acquiring gesture recognition model

You can download the GestureNet model using either wget method:

```shell
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tlt_gesturenet/versions/deployable_v1.0/zip -O tlt_gesturenet_deployable_v1.0.zip
```

or via CLI command in NGC:

```shell
ngc registry model download-version "nvidia/tlt_gesturenet:deployable_v1.0"
```

Note, as we are not retraining this model and using it directly for deployment, we seleced the `deployable_v1.0` version.


### Converting the models to TensorRT

In order to take advantage of the hardware and software accelerations on the target edge device, we want to convert our  `.etlt` models into  [TensorRT](https://developer.nvidia.com/tensorrt) engines. TensorRT is an accelerator which takes advantage of the hardware as well as allows you to modify the arithmetic precision. Changes in arithmetic precision can further increase model's throughput. 

There are two ways to convert your model into TensorRT engine. You can do it either directly via DeepStream, or by using the `tlt-converter` utility. We will show you both ways.

The trained detector model will be converted automatically by Deepstream during the first run. For the following runs you can specify the path to the produced engine in the corresponding DeepStream config. We are providing our DeepStream configs with this project.

Since the GestureNet model is quite new, the version of DeepStream we are using for this demo (5.0) does not support its conversion. However, you can convert it using the updated `tlt-converter`. To download it, click the link corresponding to your JetPack version:

* [JetPack 4.4](https://developer.nvidia.com/cuda102-trt71-jp44)
* [JetPack 4.5](https://developer.nvidia.com/cuda102-trt71-jp45)

See [this page](https://developer.nvidia.com/tlt-getting-started) if you want to use `tlt-converter` with different hardware and software.

When you have the `tlt-converter` installed on Jetson, convert the GestureNet model using the following comand:

```shell
./tlt-converter -k nvidia_tlt \ 
    -t fp16 \
    -p input_1,1x3x160x160,1x3x160x160,2x3x160x160 \
    -e /EXPORT/PATH/model.plan \
    /PATH/TO/YOUR/GESTURENET/model.etlt
```
Since we didn't change the model and are using it as it is, the model key remains the same (`nvidia_tlt `) as it is specified on the NGC.

Note, that we are converting the model into FP16 format, as we don't have any INT8 calibration file for the deployable model.

Make sure to provide correct values for your model path as well as for the export path.

### Preparing the models for deployment

There are two ways of deploying a model with DeepStream SDK: the first relies on TensorRT runtime and requires a model to be converted into a TensorRT engine, the second relies on [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). Triton Inference Server is a server which can be used as a stand-alone solution, but it can also be used integrated into DeepStream app. Such setup allows high flexibility, since it can accept models in various formats which do not necessarily have to be converted into TensorRT format. In this demo, we are showing both deployment ways by deploying our hand detector using TensorRT runtime and gesture recognition model using Triton Inference Server.

To deploy a model to DeepStream using TensorRT runtime, you only need to make sure that the model is convertable into TensorRT, which means all layers and operations in the model are supported by TensorRT. You can check the [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) for further information on supported layers and operations. 

*All models trained with Transfer Learning Toolkit are supported by TensorRT.*

### Configuring GestureNet for Triton Inference Server

In order to deploy your model via Triton Inference Server, you need to prepare a model repository in a specified format. It should have the following structure:

```
└── trtis_model_repo
    └── hcgesture_tlt
        ├── 1
        │   └── model.plan
        └── config.pbtxt
```

where `model.plan` is the `.plan` file generated with the `trt-converter`, and `config.pbtxt` has the follwing content:

```
name: "hcgesture_tlt"
platform: "tensorrt_plan"
max_batch_size: 1
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ 3, 160, 160 ]
  }
]
output [
  {
    name: "activation_18"
    data_type: TYPE_FP32
    dims: [ 6 ]
  }
]
dynamic_batching { }
```

To learn more about configuring Triton Inference Server repository, refer to [the official documentation](https://github.com/triton-inference-server/server/blob/r20.12/docs/model_repository.md).


### Customizing the deepstream-app

The sample `deepstream-app` can be configured in a flexible way: as a primary detector or a classifier, or you can even use it to cascade several models, like, for example, a detector and a classifier. In such a case, the detector will pass cropped objects of interest to the classifier. This process happens in a so called DeepStream pipeline, each component of which takes advantage of the hardware componets in Jetson devices. The pipeline for our application looks as follows:

![DeepStream pipeline](/images/ds_pipeline.png)

The GestureNet model we are using was trained on images with a big margin around the region of interest (ROI), at the same time the detector model we trained produces narrow boxes around objects of interest (hand, in our case). At first, this lead to the fact that the objects passed to the classifier were different from the representation learned by the classifier. There were two ways to solve this problem:

* either through retraining it with a new dataset reflecting our setup,
* or by extending the cropped ROIs by some suitable margin.

As we wanted to use the GestureNet model as it is, we chose the second path, which required modification of the original app. We implemented the following function to modify the meta data returned by the detetcor to crop bigger bounding boxes:

```c
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

int MARGIN = 200;

static void
modify_meta (GstBuffer * buf, AppCtx * appCtx)
{
  int frame_width;
  int frame_height;
  get_frame_width_and_height(&frame_width, &frame_height, buf);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  NvDsFrameMetaList *frame_meta_list = batch_meta->frame_meta_list;
  NvDsObjectMeta *object_meta;
  NvDsFrameMeta *frame_meta;
  NvDsObjectMetaList *obj_meta_list;
  while (frame_meta_list != NULL) {
     frame_meta = (NvDsFrameMeta *) frame_meta_list->data;
     obj_meta_list = frame_meta->obj_meta_list;
     while (obj_meta_list != NULL) {
       object_meta = (NvDsObjectMeta *) obj_meta_list->data;
       object_meta->rect_params.left = CLIP(object_meta->rect_params.left - MARGIN, 0, frame_width - 1);
       object_meta->rect_params.top = CLIP(object_meta->rect_params.top - MARGIN, 0, frame_height - 1);
       object_meta->rect_params.width = CLIP(object_meta->rect_params.left + object_meta->rect_params.width + MARGIN, 0, frame_width - 1);
       object_meta->rect_params.height = CLIP(object_meta->rect_params.top + object_meta->rect_params.height + MARGIN, 0, frame_height - 1);
       obj_meta_list = obj_meta_list->next;
     }
     frame_meta_list = frame_meta_list->next;
  }
}
```

In order to display the original bounding boxes, we implemented the following function, which restores the meta bounding boxes to their original size:

```c
static void
restore_meta (GstBuffer * buf, AppCtx * appCtx)
{

  int frame_width;
  int frame_height;
  get_frame_width_and_height(&frame_width, &frame_height, buf);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  NvDsFrameMetaList *frame_meta_list = batch_meta->frame_meta_list;
  NvDsObjectMeta *object_meta;
  NvDsFrameMeta *frame_meta;
  NvDsObjectMetaList *obj_meta_list;
  while (frame_meta_list != NULL) {
     frame_meta = (NvDsFrameMeta *) frame_meta_list->data;
     obj_meta_list = frame_meta->obj_meta_list;
     while (obj_meta_list != NULL) {
       object_meta = (NvDsObjectMeta *) obj_meta_list->data;

       // reduce the bounding boxes for output (copy the reserve value from detector_bbox_info)
       object_meta->rect_params.left = object_meta->detector_bbox_info.org_bbox_coords.left;
       object_meta->rect_params.top = object_meta->detector_bbox_info.org_bbox_coords.top;
       object_meta->rect_params.width = object_meta->detector_bbox_info.org_bbox_coords.width;
       object_meta->rect_params.height = object_meta->detector_bbox_info.org_bbox_coords.height;

       obj_meta_list = obj_meta_list->next;
     }
     frame_meta_list = frame_meta_list->next;
  }
}
```

We also implemneted this helper function to get frame width and height from the buffer.

```c
static void
get_frame_width_and_height (int * frame_width, int * frame_height, GstBuffer * buf) {
    GstMapInfo map_info;
    memset(&map_info, 0, sizeof(map_info));
    if (!gst_buffer_map (buf, &map_info, GST_MAP_READ)){
      g_print("Error: Failed to map GST buffer");
    } else {
      NvBufSurface *surface = NULL;
      surface = (NvBufSurface *) map_info.data;
      *frame_width = surface->surfaceList[0].width;
      *frame_height = surface->surfaceList[0].height;
      gst_buffer_unmap(buf, &map_info);
    }
}
```

### Building the application

To build the custom app, copy `deployment_deepstream/deepstream-app-bbox` to `/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps`.

Install the requred dependencies:

```shell
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev
```

Build an executable:

```shell
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-app-bbox
make
```

### Configuring DeepStream pipeline

Before executing the app, one has to provide configuration files. You can learn more about configuration parameters in the [official documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html). 

You can find the configuration files of this demo under `
 deployment_deepstream/egohands-deepstream-app-trtis/`. At the same directory you will also find the label files required by our models.

Finally, you need to make your models discoverable by our app. According to our configs, the directory structure under `
 deployment_deepstream/egohands-deepstream-app-trtis/` for our model storage looks as follows:

```shell
├── tlt_models
│   ├── tlt_egohands_qat
│   │   ├── calibration_qat.bin
│   │   └── resnet34_detector_qat.etlt
└── trtis_model_repo
    └── hcgesture_tlt
        ├── 1
        │   └── model.plan
        └── config.pbtxt
```

You may notice that the file `resnet34_detector_qat.etlt_b16_gpu0_int8.engine` specified in the config `config_infer_primary_peoplenet_qat.txt` is missing in the current setup. It will be generated upon the first execution and will be used directly in the following runs.

### Executing the app

In general, the execution command looks as follows:

```shell
./deepstream-app-bbox -c <config-file>
```

In our particular case with the configs provided:

```shell
./deepstream-app-bbox -c source1_primary_detector_qat.txt
```
The app should be running now!

## References

* Bambach, Sven and Lee, Stefan and Crandall, David J. and Yu, Chen (2015 December). EgoHands: A Dataset for Hands in Complex Egocentric Interactions. Retrieved February 13, 2021 from [http://vision.soic.indiana.edu/projects/egohands/](http://vision.soic.indiana.edu/projects/egohands/)

* The EgoHands conversion script was adapted from [https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/prepare_egohands.py](https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/prepare_egohands.py) by [JK Jung](https://github.com/jkjung-avt).

