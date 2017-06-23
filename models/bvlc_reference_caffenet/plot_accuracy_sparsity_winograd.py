#!/usr/bin/env python

import sys
import re

# plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

net_re = re.compile('^net:.*/train_val_(.*)\.prototxt')
model_re1 = re.compile('Finetuning from models/bvlc_reference_caffenet/[^/]*/(.*)\.caffemodel')
model_re2 = re.compile('Finetuning from models/bvlc_reference_caffenet/(.*)\.caffemodel')
model_re3 = re.compile('Resuming from models/bvlc_reference_caffenet/[^/]*/(.*)\.solverstate')
weight_decay_re = re.compile('weight_decay: (.+)')

accuracy_re = re.compile('Test net output \#\d+: accuracy = (\d+\.\d+)')
iteration_re = re.compile('Iteration (\d+), loss')
winograd_re = re.compile('Winograd Sparsity %:')
sparsity_re = re.compile('^(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)\W+(\d+\.?\d*)')

#plt.xlabel('Iteration')
#plt.ylabel('Accuracy (validation) / Sparsity')

net = None
model = None
weight_decay = None

accuracy = []
iteration = []
sparsity = [[], [], [], [], [], [], [], []]

mode = 0 # 0 : looking for accuracy, 1: looking for iteration, 2: looking for winograd, 3: looking for sparsity
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
      m = iteration_re.search(line)
      if m:
        iteration.append(int(m.group(1)))
        mode = 2
    elif mode == 2:
      m = winograd_re.search(line)
      if m:
        mode = 3
    elif mode == 3:
      m = sparsity_re.search(line)
      if m:
        for i in range(8):
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
ax1.plot(iteration[:min_len], accuracy[:min_len], label='acc', marker='*')
ax1.set_ylabel('accuracy (validation)')
ax1.set_xlabel('iteration')

ax2 = ax1.twinx()
for i in range(5):
  ax2.plot(iteration, sparsity[i][:min_len], label=('l' + str(i + 1)))
for i in range(5,8):
  ax2.plot(iteration, sparsity[i][:min_len], label=('l' + str(i + 1)), marker='+')
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=9,mode="expand", borderaxespad=0.)
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
