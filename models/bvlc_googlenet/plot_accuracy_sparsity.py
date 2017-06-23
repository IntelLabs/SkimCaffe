#!/usr/bin/env python

import sys
import re

# plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages

net_re = re.compile('^net:.*/train_val_(.*)\.prototxt')
model_re1 = re.compile('Finetuning from models/bvlc_googlenet/[^/]*/(.*)\.caffemodel')
model_re2 = re.compile('Finetuning from models/bvlc_googlenet/(.*)\.caffemodel')
model_re3 = re.compile('Resuming from models/bvlc_googlenet/[^/]*/(.*)\.solverstate')
weight_decay_re = re.compile('weight_decay: (.+)')

layer_names = [
'conv1/7x7_s2', # 1
'conv2/3x3_reduce', # 2
'conv2/3x3', # 3
'inception_3a/1x1', # 4
'inception_3a/3x3_reduce', # 5
'inception_3a/3x3', # 6
'inception_3a/5x5_reduce', # 7
'inception_3a/5x5', # 8
'inception_3a/pool_proj', # 9
'inception_3b/1x1', # 10
'inception_3b/3x3_reduce', # 11
'inception_3b/3x3', # 12
'inception_3b/5x5_reduce', # 13
'inception_3b/5x5', # 14
'inception_3b/pool_proj', # 15
'inception_4a/1x1', # 16
'inception_4a/3x3_reduce', # 17
'inception_4a/3x3', # 18
'inception_4a/5x5_reduce', # 19
'inception_4a/5x5', # 20
'inception_4a/pool_proj', # 21
'loss1/conv', # 22
'loss1/fc', # 23 fc
'loss1/classifier', # 24 fc
'inception_4b/1x1', # 25
'inception_4b/3x3_reduce', # 26
'inception_4b/3x3', # 27
'inception_4b/5x5_reduce', # 28
'inception_4b/5x5', # 29
'inception_4b/pool_proj', # 30
'inception_4c/1x1', # 31
'inception_4c/3x3_reduce', # 32
'inception_4c/3x3', # 33
'inception_4c/5x5_reduce', # 34
'inception_4c/5x5', # 35
'inception_4c/pool_proj', # 36
'inception_4d/1x1', # 37
'inception_4d/3x3_reduce', # 38
'inception_4d/3x4', # 39
'inception_4d/5x5_reduce', # 40
'inception_4d/5x5', # 41
'inception_4d/pool_proj', # 42
'loss2/conv', # 43
'loss2/fc', # 44 fc
'loss2/classifier', # 45 fc
'inception_4e/1x1', # 46
'inception_4e/3x3_reduce', # 47
'inception_4e/3x3', # 48
'inception_4e/5x5_reduce', # 49
'inception_4e/5x5', # 50
'inception_4e/pool_proj', # 51
'inception_5a/1x1', # 52
'inception_5a/3x3_reduce', # 53
'inception_5a/3x3', # 54
'inception_5a/5x5_reduce', # 55
'inception_5a/5x5', # 56
'inception_5a/pool_proj', # 57
'inception_5b/1x1', # 58
'inception_5b/3x3_reduce', # 59
'inception_5b/3x3', # 60
'inception_5b/5x5_reduce', # 61
'inception_5b/5x5', # 62
'inception_5b/pool_proj', # 63
'loss3/classifier', # 64 fc
]

accuracy_re = re.compile('Test net output \#\d+: loss3/top-1 = (\d+\.\d+)')
accuracy_top5_re = re.compile('Test net output \#\d+: loss3/top-5 = (\d+\.\d+)')
iteration_re = re.compile('Iteration (\d+), loss')

sparsity_re_string = '^'
for i in range(len(layer_names)):
  sparsity_re_string += '(\d+\.?\d*)\W+'

sparsity_re = re.compile(sparsity_re_string)
#sparsity_re = re.compile('^(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+(\d+\.?\d*)\W+\d+\.?\d*\W+')

#plt.xlabel('Iteration')
#plt.ylabel('Accuracy (validation) / Sparsity')

net = None
model = None
weight_decay = None

accuracy = []
accuracy_top5 = []
iteration = []
sparsity = []
for i in range(len(layer_names)):
  sparsity.append([])

mode = 0 # 0 : looking for accuracy, 1: looking for iteration, 2: looking for sparsity
with open(sys.argv[1] + "/train.info") as logfile:
  for line in logfile:
    m = net_re.search(line)
    if m:
      net = m.group(1)

    m = model_re1.search(line)
    if m:
      model = m.group(1)
    else:
      m = model_re2.search(line)
      if m:
        model = m.group(1)
      else:
        m = model_re3.search(line)
        if m:
          model = m.group(1)

    m = weight_decay_re.search(line)
    if m:
      weight_decay = m.group(1)

    if mode == 0:
      m = accuracy_re.search(line)
      if m:
        accuracy.append(float(m.group(1)))
        mode = 1
    elif mode == 1:
      m = accuracy_top5_re.search(line)
      if m:
        accuracy_top5.append(float(m.group(1)))
        mode = 2
    elif mode == 2:
      m = iteration_re.search(line)
      if m:
        iteration.append(int(m.group(1)))
        mode = 3
    elif mode == 3:
      m = sparsity_re.search(line)
      if m:
        for i in range(len(layer_names)):
          sparsity[i].append(float(m.group(i + 1))/100)
        mode = 0

#print len(iteration)
#print len(accuracy)
#for i in range(8):
  #print len(sparsity[i])

#for i in range(len(accuracy)):
  #print iteration[i], accuracy[i],
  #for j in range(8):
    #print sparsity[j][i],
  #print
min_len = min(len(iteration), len(accuracy))

fig, ax1 = plt.subplots()
ax1.plot(iteration[:min_len], accuracy[:min_len], label='top1_acc', marker='*')
ax1.plot(iteration[:min_len], accuracy_top5[:min_len], label='top5_acc', marker='*')
ax1.set_ylabel('accuracy (validation)')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=9,mode="expand", borderaxespad=0.)
ax1.set_xlabel('iteration')

ax2 = ax1.twinx()
viridis = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=len(layer_names))
scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=viridis)
for i in range(len(layer_names)):
  ax2.plot(iteration, sparsity[i][:min_len], label=layer_names[i], color=scalarMap.to_rgba(i))
#ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=9,mode="expand", borderaxespad=0.)
ax2.set_ylabel('sparsity')
#plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
#ax = plt.subplot(111)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
#ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.6))

#if len(sys.argv) > 1
  #outfile = sys.argv[-1]
#else
#outfile = net + "_" + model + "_" + weight_decay + ".png"
outfile = sys.argv[1] + ".png"
plt.savefig(outfile)
