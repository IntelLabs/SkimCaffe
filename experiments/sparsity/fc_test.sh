#!/bin/bash

OMP_NUM_THREADS=32 ./fc_test ~/packages/caffe/matrices/fc_0.1_ft_caffenet_0.57368_5e-05/fc6.mtx

OMP_NUM_THREADS=1 ./fc_test lstm.mtx 32

OMP_NUM_THREADS=4 ./fc_test lstm.mtx 256

OMP_NUM_THREADS=11 ./fc_test lstm.mtx 32

OMP_NUM_THREADS=32 ./fc_test lstm.mtx
OMP_NUM_THREADS=32 ./fc_test lstm.mtx 32
OMP_NUM_THREADS=32 ./fc_test lstm.mtx 256 32

OMP_NUM_THREADS=44 ./fc_test lstm.mtx
OMP_NUM_THREADS=44 ./fc_test lstm.mtx 32
OMP_NUM_THREADS=44 ./fc_test lstm.mtx 64
OMP_NUM_THREADS=44 ./fc_test lstm.mtx 256 32
