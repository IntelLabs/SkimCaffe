## SkimCaffe Specific Description

SkimCaffe has been only tested with bvlc_reference_caffenet bvlc_googlenet, and
there could be places where things do not work if you use other networks.
Please let us know if you encounter such issues and share .prototxt of the
network you are using.
We will try our best to support it as well.
Eventually, our sparse CNN implementation should be general enough to handle
all kinds of networks.

We assume you have a recent Intel compiler and MKL installed.
Tested environments: (Intel compiler version 15.0.3.187 and boost 1.59.0)
We also assume you have a recent x86 CPU with AVX2 or AVX512 support.
Direct sparse convolution and sparse fully-connected layers is only tested for AlexNet.
More details on direct sparse convolution is described at: https://arxiv.org/abs/1608.01409

1) Set up Intel compiler environment (compilervars.sh or compilervars.csh)

2) Compile SpMP:

```
cd src/SpMP
make
cd ../libxsmm
make
```

3) Build Caffe as usual

4) Test:

```
bzip2 -d models/bvlc_reference_caffenet/fc_0.1_ft_caffenet_0.57368_5e-05.caffemodel.bz2
env OMP_NUM_THREADS=16 KMP_AFFINITY=granularity=fine,compact,1 build/tools/caffe.bin test -model models/bvlc_reference_caffenet/test_direct_sconv.prototxt -weights models/bvlc_reference_caffenet/fc_0.1_ft_caffenet_0.57368_5e-05.caffemodel -iterations 3 # assume you have 16 cores. Adjust OMP_NUM_THREADS variable accordingly for your number of cores
env OMP_NUM_THREADS=16 KMP_AFFINITY=granularity=fine,compact,1 build/tools/caffe.bin test -model models/bvlc_googlenet/test_direct_sconv.prototxt -weights models/bvlc_googlenet/caffenet_train_iter_273309.caffemodel -iterations 3 # test with GoogLeNet
```

Example output from Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz
Used 88 omp threads (using hyper-threading) and batch size 264

```
I1005 14:22:20.452144 68219 conv_relu_pool_lrn_layer.cpp:747] conv1 K-cycles-per-file max 752.933 avg 508.823 mFlops-per-file 210.83 GF/s 614.611
I1005 14:22:20.489462 68219 conv_relu_pool_lrn_layer.cpp:747] conv2 K-cycles-per-file max 226.008 avg 220.347 mFlops-per-file 447.898 GF/s 4349.88
I1005 14:22:20.503149 68219 conv_relu_layer.cpp:227] conv3 K-cycles-per-file max 81.9624 avg 77.7974 mFlops-per-file 299.041 GF/s 8008.27
I1005 14:22:20.516557 68219 conv_relu_layer.cpp:227] conv4 K-cycles-per-file max 75.9429 avg 69.7938 mFlops-per-file 224.281 GF/s 6482.28
I1005 14:22:20.532255 68219 conv_relu_pool_layer.cpp:366] conv5 K-cycles-per-file max 65.375 avg 60.152 mFlops-per-file 149.52 GF/s 5020.09
I1005 14:22:20.540031 68219 inner_product_relu_dropout_layer.cpp:386] csrmm takes 0.00632691 effective GF/s 3150.25 real GF/s 412.774
I1005 14:22:20.543900 68219 inner_product_relu_dropout_layer.cpp:386] csrmm takes 0.00382113 effective GF/s 2318.26 real GF/s 460.349
I1005 14:22:20.571082 68219 net.cpp:629]  Test time of data	80.543 ms ( 26.7871 % )
I1005 14:22:20.571113 68219 net.cpp:629]  Test time of label_data_1_split	0.005 ms ( 0.00166291 % )
I1005 14:22:20.571120 68219 net.cpp:629]  Test time of conv1	101.675 ms ( 33.8152 % )
I1005 14:22:20.571126 68219 net.cpp:629]  Test time of conv2	37.31 ms ( 12.4086 % )
I1005 14:22:20.571132 68219 net.cpp:629]  Test time of conv3	13.654 ms ( 4.54107 % )
I1005 14:22:20.571137 68219 net.cpp:629]  Test time of conv4	13.459 ms ( 4.47622 % )
I1005 14:22:20.571143 68219 net.cpp:629]  Test time of conv5	15.621 ms ( 5.19526 % )
I1005 14:22:20.571148 68219 net.cpp:629]  Test time of fc6	7.788 ms ( 2.59015 % )
I1005 14:22:20.571153 68219 net.cpp:629]  Test time of fc7	5.858 ms ( 1.94826 % )
I1005 14:22:20.571158 68219 net.cpp:629]  Test time of fc8	3.494 ms ( 1.16204 % )
I1005 14:22:20.571163 68219 net.cpp:629]  Test time of fc8_fc8_0_split	0.002 ms ( 0.000665163 % )
I1005 14:22:20.571168 68219 net.cpp:629]  Test time of accuracy	2.019 ms ( 0.671482 % )
I1005 14:22:20.571174 68219 net.cpp:629]  Test time of loss	19.25 ms ( 6.4022 % )
I1005 14:22:20.571182 68219 caffe.cpp:294] Total forwarding time: 300.678 ms
I1005 14:22:20.571190 68219 caffe.cpp:297] Loss: 1.67216
I1005 14:22:20.571216 68219 caffe.cpp:309] accuracy = 0.593434
I1005 14:22:20.571239 68219 caffe.cpp:309] loss = 1.67216 (* 1 = 1.67216 loss)
I1005 14:22:20.571251 68219 caffe.cpp:314] Total-images-processed: 792
I1005 14:22:20.571255 68219 caffe.cpp:317] conv1 K-cycles-per-file 659.516 mFlops-per-file 210.83 GF/s 701.666
I1005 14:22:20.571269 68219 caffe.cpp:317] conv2 K-cycles-per-file 258.566 mFlops-per-file 447.898 GF/s 3802.15
I1005 14:22:20.571277 68219 caffe.cpp:317] conv3 K-cycles-per-file 96.358 mFlops-per-file 299.041 GF/s 6811.84
I1005 14:22:20.571285 68219 caffe.cpp:317] conv4 K-cycles-per-file 78.07 mFlops-per-file 224.281 GF/s 6305.59
I1005 14:22:20.571291 68219 caffe.cpp:317] conv5 K-cycles-per-file 64.486 mFlops-per-file 149.52 GF/s 5089.22
I1005 14:22:20.571300 68219 caffe.cpp:317] fc6 K-cycles-per-file 59.945 mFlops-per-file 75.4975 GF/s 2764.4
I1005 14:22:20.571306 68219 caffe.cpp:317] fc7 K-cycles-per-file 31.91 mFlops-per-file 33.5544 GF/s 2308.03
```

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
