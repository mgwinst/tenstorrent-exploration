#!/bin/bash
export TT_METAL_HOME=/home/matteo/tenstorrent/tt-metal
clang++ -std=c++20 matmul.cpp -stdlib=libc++ -I$TT_METAL_HOME -I/$TT_METAL_HOME/tt_metal -I$TT_METAL_HOME/tt_metal/third_party/umd \
    -I$TT_METAL_HOME/tt_metal/hw/inc/grayskull -I$TT_METAL_HOME/tt_metal/hw/inc \
    -L$TT_METAL_HOME/build/lib -ltt_metal
TT_METAL_DPRINT_CORES=0,0 LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib/ ./a.out
