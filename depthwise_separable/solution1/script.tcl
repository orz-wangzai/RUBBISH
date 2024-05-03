############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project depthwise_separable
set_top depthwise_separable
add_files depthwise_separable/bn_qrelu2d.h
add_files depthwise_separable/conv2d.h
add_files depthwise_separable/function.h
add_files depthwise_separable/matrix_vector_unit.h
add_files depthwise_separable/sliding_window_unit.h
add_files depthwise_separable/stream_tools.h
add_files -tb depthwise_separable/conv_test.cpp
open_solution "solution1"
set_part {xczu2eg-sfvc784-2lv-e} -tool vivado
create_clock -period 10 -name default
#source "./depthwise_separable/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
