#!/bin/bash
set -e
set -x

folder="models/vgg"
file_prefix="caffenet_train"
model_path="models/vgg"

if [ "$#" -lt 7 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr weight_decay prune_threshold max_threshold_factor winograd_sparsity_factor ic_decay oc_decay kernel_decay device_id template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
weight_decay=$2
prune_threshold=$3
max_threshold_factor=$4
winograd_sparsity_factor=$5
ic_decay=$6
oc_decay=$7
kernel_decay=$8
solver_mode="GPU"
device_id=0

current_time=$(date +%Y-%m-%d-%H-%M-%S)
#current_time=$(date)
#current_time=${current_time// /_}
#current_time=${current_time//:/-}

snapshot_name=${base_lr}_${weight_decay}_${prune_threshold}_${max_threshold_factor}_${winograd_sparsity_factor}_${ic_decay}_${oc_decay}_${kernel_decay}_${current_time}
snapshot_path=$folder/$snapshot_name
mkdir $snapshot_path
echo $@ > $snapshot_path/cmd.log

solverfile=$snapshot_path/solver.prototxt
template_file='template_solver.prototxt'
#if [ "$#" -ge 7 ]; then
template_file=${10}
#fi

cat $folder/${template_file} > $solverfile
echo "block_group_decay: $kernel_decay" >> $solverfile
echo "kernel_shape_decay: $ic_decay" >> $solverfile
echo "breadth_decay: $oc_decay" >> $solverfile
echo "winograd_sparsity_factor: $winograd_sparsity_factor" >> $solverfile
echo "prune_threshold: $prune_threshold" >> $solverfile
echo "max_threshold_factor: $max_threshold_factor" >> $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
#if [ "$#" -ge 6 ]; then
if [ "$9" -ne "-1" ]; then
	device_id=$9
	#echo "device_id: $device_id" >> $solverfile
  echo $snapshot_name > $folder/$device_id
else
	solver_mode="CPU"
fi
#fi
echo "solver_mode: $solver_mode" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

if [ "$#" -ge 11 ]; then
	tunedmodel=${11}
	file_ext=$(echo ${tunedmodel} | rev | cut -d'.' -f 1 | rev)
	if [ "$file_ext" = "caffemodel" ]; then
    if [ "$9" -ne "-1" ]; then
	    ./build/tools/caffe.bin train -gpu $device_id --solver=$solverfile --weights=$model_path/$tunedmodel  > "${snapshot_path}/train.info" 2>&1
    else
	    ../caffe_scnn_cpu_only/build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel  > "${snapshot_path}/train.info" 2>&1
    fi
	else
	  ./build/tools/caffe.bin train -gpu $device_id --solver=$solverfile --snapshot=$model_path/$tunedmodel > "${snapshot_path}/train.info" 2>&1
	fi
else
	./build/tools/caffe.bin train -gpu $device_id --solver=$solverfile   > "${snapshot_path}/train.info" 2>&1
fi

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info

#cd $folder
#finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
#for file in $finalfiles; do
#	cp $file "$current_time-$file"
#done

content="$(hostname) done: ${0##*/} ${@}. Results in ${snapshot_path}"
echo ${content} | mail -s "Training done" jongsoo.park@intel.com
