# Mannequin Challenge Code and Trained Models

This repository contains inference code for models trained on the Mannequin
Challenge dataset introduced in the CVPR 2018 paper "Learning the Depths of Moving People by Watching Frozen People."

## Setup

The code is based on PyTorch. The code has been tested with PyTorch 1.1 and Python 3.6. 

We recommend setting up a `virtualenv`
environment for installing PyTorch and the other necessary Python packages. The [TensorFlow installation
guide](https://www.tensorflow.org/install/pip) may be helpful (follow steps 1
and 2) or follow the `virtualenv` documentation.

Once your environment is set up and activated, install the necessary packages:

```
(pytorch)$ pip install torch torchvision scikit-image h5py
```

The model checkpoints are stored on Google Cloud and may be retrieved by running:

```
(pytorch)$ ./fetch_checkpoints.sh
```

## Single-View Inference

Our test set for single-view inference is the DAVIS dataset (cite). Download and unzip it by running

```
(pytorch)$ ./fetch_davis_data.sh
```

Then run the DAVIS inference script:

```
(pytorch)$ python test_davis_videos.py --input_nc=3
```

The `input_nc` flag sets the number of input channels (here RGB) for the network to expect. Once the run completes, visualizations of the output should be available in `test_data/viz_predictions`.

## Full Model Inference

The full model described in the paper requires several additional inputs: the human segmenation mask, the depth-from-parallax, and (optionall) a human keypoint buffer. We provide a preprocessed version of the TUM RGBD data (cite) that includes these inputs. Download (~8GB) and unzip it using the script:

```
(pytorch)$ ./fetch_tum_data.sh
```

To reproduce the numbers in Table XXX of the paper, run:

```
(pytorch)$ python test_tum.py --input_nc=7
```

The script prints running averages of the various error metrics as it runs. When the script completes, the final error metrics are shown.


## Acknowledgements

If you find the code or results useful, please cite:

XXXXXX


Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
