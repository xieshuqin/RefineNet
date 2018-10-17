# Install RefineNet 

Our code is tested on caffe2 built with cuda9.0 and cudnn 7.2. We suggest using docker for maintaining the dependency. 

## Data folder
First we prepare a data folder for the docker to interact with the server(retriving datasets and saving results). 
```
mkdir -p /path/to/data
mkdir -p /path/to/data/loggings/RefineNet
mkdir -p /path/to/data/experiments/RefineNet
```

The `data` folder should have the following structure. 
```
data
|_ models
   |_ R-50.pkl
   |_ R-101.pkl
   |_ ...
|_ loggings
   |_ RefineNet
   |_ ...
|_ experiments
   |_ RefineNet
   |_ ...
|_ coco
   |_ coco_train2014
   |  |_ <im-1-name>.jpg
   |  |_ ...
   |  |_ <im-N-name>.jpg
   |_ coco_val2014
   |_ ...
   |_ annotations
      |_ instances_train2014.json
      |_ ...
```

## Caffe2 

First load RefineNet docker image
``` 
docker load < refinenet.tar
```

Then run 
```
nvidia-docker run -it -v /path/to/data:/root/data --name test2 refinenet:latest /bin/sh
```

It creates a container named test2. After the experiments are done, we can escape by using `Ctrl-p + Ctrl-q`, which escapes the container without shutting it down. To attach to the same container again, use `docker attach test2` . 

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

**IMPORTANT**
Don't use `Ctrl-d` when escaping the docker. It will shut down the docker so the programs are terminated.

## Other Dependencies

Install Python dependencies:

```
pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock
```

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## RefineNet
Clone the RefineNet repo
```
# REFINENET=/path/to/RefineNet
git clone https://github.com/xieshuqin/RefineNet.git $REFINENET
```

Set up customized caffe2 ops and remake caffe2
``` 
# CAFFE2=/path/to/caffe2
cp $REFINENET/lib/ops/caffe2_ops/* $CAFFE2/caffe2/operators/
cd $CAFFE2/build && cmake ..
make install -j 
```

Set up Python modules:

```
cd $DETECTRON/lib && make
```

Now we are good to go!


## Running experiments
For mask tasks
``` 
python2 tools/train_net.py --cfg configs/masks/e2e_mask_refine_R-50-FPN_1x.yaml --multi-gpu-testing OUTPUT_DIR ~/data/experiments/RefineNet/your_exp_name 2>&1 | tee ~/data/loggings/RefineNet/your_exp_name.log
``` 

For keypoints tasks
``` 
python2 tools/train_net.py --cfg configs/keypoints/e2e_keypoint_refine_R-50-FPN_1x.yaml --multi-gpu-testing OUTPUT_DIR ~/data/experiments/RefineNet/your_exp_name 2>&1 | tee ~/data/loggings/RefineNet/your_exp_name.log
```

For using the GenerateIndicatorOp implemented in CUDA, add `REFINENET.USE_CUDA_INDICATOR_OP True` to the command. This should be faster than the PythonOp. However, the performance is not guarantee to be exactly the same as the PythonOp. Will be update later when the performances are the same.

To training with different number of GPUS, please use the linear scaling rule and follow the instructions in [`GETTING_STARTED.md`](GETTING_STARTED.md) to change the learning rate and total iterations.