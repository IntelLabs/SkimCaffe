## SkimCaffe Specific Description

> :warning: **DISCONTINUATION OF PROJECT** - 
> *This project will no longer be maintained by Intel.
> Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.*
> **Intel no longer accepts patches to this project.**
> *If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.*


A Caffe branch for training sparse CNN that provides 80-95% sparsity in
convolution and fully-connected layers (tested with AlexNet, GoogLeNet-v1, and
Resnet-50).
Our optimized sparse convolution and sparse-matrix-times-dense-matrix routines
effectively take advantage of the sparsity achieving ~3x speedups with 90%
sparsity (our sparse convolution routine will be migrated to libxsmm library so
that it can be also used by other frameworks like TensorFlow).
SkimCaffe also has experimental features of getting sparsity in Winograd
(L1_Winograd regularization) and we have prelimiary results for AlexNet.
Please let us know if you're interested in this experimental feature.
More details are described in the following papers:

- https://arxiv.org/abs/1608.01409 (Holistic Sparse CNN: Forging the Trident of Accuracy, Speed, and Size, Jongsoo Park, Sheng Li, Wei Wen, Hai Li, Yiran Chen, and Pradeep Dubey)
- https://openreview.net/forum?id=rJPcZ3txx (updated version of the above paper accepted to ICLR'17, Faster CNNs with Direct Sparse Convolutions and Guided Pruning, Jongsoo Park, Sheng Li, Wei Wen, Ping Tak Peter Tang, Hai Li, Yiran Chen, and Pradeep Dubey)
- https://arxiv.org/abs/1702.08597 (Enabling Sparse Winograd Convolution by Native Pruning, Sheng Li, Jongsoo Park, and Ping Tak Peter Tang)

Sparsity of pruned models:
```
CaffeNet (a small variation of AlexNet):
models/bvlc_reference_caffenet/logs/acc_57.5_0.001_5e-5_ft_0.001_5e-5/0.001_5e-05_0_1_0_0_0_0_Sun_Jan__8_07-35-54_PST_2017/caffenet_train_iter_640000.caffemodel.bz2
top-1 accuracy: 0.57478
conv2: sparsity 85.6455%
conv3: 93.1
conv4: 91.808
conv5: 88.4903
fc6: 90.2809
fc7: 84.3741
fc8: 73.8236

GoogLeNet
models/bvlc_googlenet/gesl_0.686639_0.001_0.00005_ft_0.001_0.0001.caffemodel.bz2
top-1 accuracy: 0.686639
top-5 accuracy: 0.886302
inception_3a/5x5    84.5625
inception_3b/5x5    88.5482
inception_4a/5x5    83.7604
inception_4b/5x5    87.3958
inception_4c/5x5    88.0052
inception_4d/5x5    90.3867
inception_4e/5x5    94.416
inception_5a/5x5    93.3984
inception_5b/5x5    94.8717
conv2/3x3           81.6533
inception_3a/3x3    90.0861
inception_3b/3x3    90.4699
inception_4a/3x3    87.968
inception_4b/3x3    96.1057
inception_4c/3x3    94.8968
inception_4d/3x3    92.4134
inception_4e/3x3    98.6999
inception_5a/3x3    96.2437
inception_5b/3x3    94.9538
loss1/fc            97.9414
loss1/classifier    93.5289
loss2/fc            98.1798
loss2/classifier    94.1971
loss3/classifier    90.6519

Resnet-50
models/resnet/caffenet_train_iter_2000000.solverstate.bz2
top-1 accuracy: 0.72838
top-5 accuracy: 0.913282

res2a_branch2b      88.0046
res2b_branch2b      83.9708
res2c_branch2b      80.7943
res3a_branch2b      91.7562
res3b_branch2b      91.9101
res3c_branch2b      95.5024
res3d_branch2b      91.1343
res4a_branch2b      96.5534
res4b_branch2b      97.4677
res4c_branch2b      97.1471
res4d_branch2b      96.7648
res4e_branch2b      96.8435
res4f_branch2b      96.8374
res5a_branch2b      97.7809
res5b_branch2b      98.0271
res5c_branch2b      97.9203
fc1000              91.5099
```

SkimCaffe has been only tested with bvlc_reference_caffenet, bvlc_googlenet,
and resnet, and there could be places where things do not work if you use other
networks.
Please let us know if you encounter such issues and share .prototxt of the
network you are using.
We will try our best to support it as well.
Eventually, our sparse CNN implementation should be general enough to handle
all kinds of networks.

We assume you have a recent Intel compiler and MKL installed.
Tested environments: Intel compiler version 15.0.3.187 or newer. boost 1.59.0 . MKL 2017 or newer to use MKL-DNN.
Direct sparse convolution and sparse fully-connected layers is only tested for AlexNet and GoogLeNet.

Build instruction:

1) Set up Intel compiler environment (compilervars.sh or compilervars.csh)

```
/opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 # assuming Intel compiler is installed under /opt/intel and you're using BASH
```

2) Create Makefile.config based on Makefile.config.example

```
cp Makefile.config.example Makefile.config
# Then, change Makefile.config if needed. For example, you need to add boost include path to INCLUDE_DIRS and boost library path to LIBRARY_DIRS
```

3) Build libxsmm

```
make libxsmm
```

4) Build Caffe as usual

Additional options:
AVX=3 # compile for Skylake
AVX=3 MIC=1 # compile for Knights Landing
AVX=2 # compiles for AVX2

5) Test:

```
# Test sparse CaffeNet
bzip2 -d models/bvlc_reference_caffenet/logs/acc_57.5_0.001_5e-5_ft_0.001_5e-5/0.001_5e-05_0_1_0_0_0_0_Sun_Jan__8_07-35-54_PST_2017/caffenet_train_iter_640000.caffemodel.bz2
export OMP_NUM_THREADS=16 # assume you have 16 cores. Adjust OMP_NUM_THREADS variable accordingly for your number of cores
export KMP_AFFINITY=granularity=fine,compact,1
build/tools/caffe.bin test -model models/bvlc_reference_caffenet/test_direct_sconv_mkl.prototxt -weights models/bvlc_reference_caffenet/logs/acc_57.5_0.001_5e-5_ft_0.001_5e-5/0.001_5e-05_0_1_0_0_0_0_Sun_Jan__8_07-35-54_PST_2017/caffenet_train_iter_640000.caffemodel

# Test sparse GoogLeNet
bzip2 -d models/bvlc_googlenet/gesl_0.686639_0.001_0.00005_ft_0.001_0.0001.caffemodel.bz2
build/tools/caffe.bin test -model models/bvlc_googlenet/test_direct_sconv.prototxt -weights models/bvlc_googlenet/gesl_0.686639_0.001_0.00005_ft_0.001_0.0001.caffemodel
```

Example output from Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz
Used 88 omp threads (using hyper-threading) and batch size 264

```
I0112 19:44:17.246243 83804 conv_relu_pool_lrn_layer.cpp:749] conv2 K-cycles-per-file max 203.198 avg 201.927 mFlops-per-file 447.898 GF/s 4838.16
I0112 19:44:17.255386 83804 conv_relu_layer.cpp:217] conv3 wall clock-time 0.00903296 padding-time 0.000645399 relu-time 9.89437e-05
I0112 19:44:17.255430 83804 conv_relu_layer.cpp:227] conv3 K-cycles-per-file max 68.5256 avg 66.8225 mFlops-per-file 299.041 GF/s 9578.53
I0112 19:44:17.263487 83804 conv_relu_layer.cpp:217] conv4 wall clock-time 0.00803304 padding-time 0.00048995 relu-time 0.000106096
I0112 19:44:17.263531 83804 conv_relu_layer.cpp:227] conv4 K-cycles-per-file max 59.3055 avg 56.6332 mFlops-per-file 224.281 GF/s 8300.77
I0112 19:44:17.273319 83804 conv_relu_pool_layer.cpp:342] conv5 wall clock-time 0.009758 pool-time 0.00255013
I0112 19:44:17.273363 83804 conv_relu_pool_layer.cpp:368] conv5 K-cycles-per-file max 52.2832 avg 51.2715 mFlops-per-file 149.52 GF/s 6277.1
I0112 19:44:17.279690 83804 inner_product_relu_dropout_layer.cpp:228] csrmm takes 0.00629187 effective GF/s 3167.79 real GF/s 308.281
I0112 19:44:17.284083 83804 inner_product_relu_dropout_layer.cpp:228] csrmm takes 0.00371099 effective GF/s 2387.07 real GF/s 373.345
I0112 19:44:17.286274 83804 inner_product_layer.cpp:223] csrmm takes 0.00164509 effective GF/s 1314.63 real GF/s 344.16
I0112 19:44:17.289224 83804 net.cpp:654]  Test time of data     29.689 ms ( 16.2876 % )
I0112 19:44:17.289305 83804 net.cpp:654]  Test time of label_data_1_split       0.002 ms ( 0.00109721 % )
I0112 19:44:17.289350 83804 net.cpp:654]  Test time of conv1    60.57 ms ( 33.2291 % )
I0112 19:44:17.289393 83804 net.cpp:654]  Test time of relu1    5.189 ms ( 2.84672 % )
I0112 19:44:17.289409 83804 net.cpp:654]  Test time of pool1    5.894 ms ( 3.23349 % )
I0112 19:44:17.289482 83804 net.cpp:654]  Test time of norm1    6.289 ms ( 3.45019 % )
I0112 19:44:17.289535 83804 net.cpp:654]  Test time of conv2    31.84 ms ( 17.4676 % )
I0112 19:44:17.289557 83804 net.cpp:654]  Test time of conv3    9.093 ms ( 4.98848 % )
I0112 19:44:17.289572 83804 net.cpp:654]  Test time of conv4    8.092 ms ( 4.43932 % )
I0112 19:44:17.289587 83804 net.cpp:654]  Test time of conv5    9.819 ms ( 5.38677 % )
I0112 19:44:17.289603 83804 net.cpp:654]  Test time of fc6      6.97 ms ( 3.82379 % )
I0112 19:44:17.289619 83804 net.cpp:654]  Test time of fc7      4.253 ms ( 2.33322 % )
I0112 19:44:17.289638 83804 net.cpp:654]  Test time of fc8      1.743 ms ( 0.956221 % )
I0112 19:44:17.289654 83804 net.cpp:654]  Test time of fc8_fc8_0_split  0.001 ms ( 0.000548607 % )
I0112 19:44:17.289669 83804 net.cpp:654]  Test time of accuracy 2.192 ms ( 1.20255 % )
I0112 19:44:17.289685 83804 net.cpp:654]  Test time of loss     0.644 ms ( 0.353303 % )
I0112 19:44:17.289746 83804 caffe.cpp:330] Total forwarding time: 182.28 ms
I0112 19:44:17.660748 83804 caffe.cpp:333] Loss: 1.83185
I0112 19:44:17.660802 83804 caffe.cpp:345] accuracy = 0.573712
I0112 19:44:17.660838 83804 caffe.cpp:345] loss = 1.83185 (* 1 = 1.83185 loss)
I0112 19:44:17.660853 83804 caffe.cpp:350] Total-images-processed: 13200
I0112 19:44:17.660862 83804 caffe.cpp:353] conv2 K-cycles-per-file 206.582 mFlops-per-file 447.898 GF/s 4758.9
I0112 19:44:17.660883 83804 caffe.cpp:353] conv3 K-cycles-per-file 74.764 mFlops-per-file 299.041 GF/s 8779.24
I0112 19:44:17.660899 83804 caffe.cpp:353] conv4 K-cycles-per-file 65.242 mFlops-per-file 224.281 GF/s 7545.37
I0112 19:44:17.660913 83804 caffe.cpp:353] conv5 K-cycles-per-file 57.779 mFlops-per-file 149.52 GF/s 5679.97
I0112 19:44:17.660930 83804 caffe.cpp:353] fc6 K-cycles-per-file 55.684 mFlops-per-file 75.4975 GF/s 2975.93
I0112 19:44:17.660943 83804 caffe.cpp:353] fc7 K-cycles-per-file 34.338 mFlops-per-file 33.5544 GF/s 2144.81
I0112 19:44:17.660958 83804 caffe.cpp:353] fc8 K-cycles-per-file 14.645 mFlops-per-file 8.192 GF/s 1227.76
```

## How to prune

1) Pruning

CaffeNet, GoogLeNet-v1, and Resnet-50 already have scripts that generate solver
prototxt for pruning/fine-tuning and run Caffe.
You can create your own script for you network or you can look at the script to
see how you need to change the solver prototxt.

You also need to change regularization type to L1 for the conv/fc layers you want to prune.
Please see models/bvlc_reference_caffenet/train_val_convl1l2.prototxt for example.
If you want to play with pruning in Winograd (more experimental feature), you can
try L1_Winograd regularization type.

Then, you can start pruning. For example, the command below will prune the
reference caffenet model with base learning rate 0.001, weight decay 5e-5,
threshold 0.0001, and threshold adjusting factor 10.
We found this learning rate and weight decay in general works well.
The higher the weight decay, you will get higher sparsity but with lower accuracy.
Threshold 0.0001 and threshold adjusting factor 10 means that, when a gradient
respect to a parameter is 1, we threshold the parameter to 0 if its absolute
value is less than or equal to 0.0001
If the gradient with respect to another parameter is bigger, we apply
thresholding more conservatively using a threshold smaller than 0.0001.
The threshold adjusting factor determines how aggressively we adjust the
threshold based on the gradients.
We found 0.0001 threshold and adjusting factor 10 works well in general.
We suggest to mostly change weight decay to control sparsity vs. accuracy
trade-off instead of playing with the threshold related parameters.
You can ignore the following 4 command line arguments of 0, which are used
for group-wise sparsity. If you're interested, you can look at Wei Wen's
Caffe branch.
The last numeric parameter is GPU id you want to use.
The next parameter is the template solver prototxt.
You only need to specify net prototxt, test_iter, test_interval, lr_policy,
display, max_iter, regularization_type, and snapshot in the template solver
prototxt.
Other parameters like base learning rate and weight decay are set by the
script, which will generate another solver prototxt and run Caffe using the
generated solver prototxt.
The script will create a directory whose name is generated using the learning
rate, weight decay, dates, and so on.
The script will create the solver prototxt inside the directory and store
snapshots and caffemodels there.

```
models/bvlc_reference_caffenet/train_script.sh 0.001 5e-5 0.0001 10 0 0 template_l1_solver.prototxt caffenet_0.57368.caffemodel
```

Over the course of training, you may want to look at the progress of sparsity
and accuracy.
You can use plot_accuracy_sparsity.py (the only command line argument is the
directory that train_script.sh created).
This Python script will generate an image file that plots showing how the
accuracy and sparsity change over time.

2) Fine-tuning

You can use the same train_script.sh but with a different solver prototxt where
you use L2 regularization and fix the sparsity by specifying DISCONNECT_ELTWISE
connectivity mode (see models/bvlc_reference_caffenet/train_val_ft.prototxt for
example).
You start from the caffemodel generated by the above pruning process.

```
models/bvlc_reference_caffenet/train_script.sh 0.001 5e-5 0 0 0 0 template_finetune_solver.prototxt caffenet_pruned.caffemodel
```

In models/bvlc_reference_caffenet , models/bvlc_googlenet , and models/resnet
directories, log subdirectory shows how I pruned these models.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
