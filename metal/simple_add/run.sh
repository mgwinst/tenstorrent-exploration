#!/bin/bash
export TT_METAL_HOME=/home/matteo/tenstorrent/tt-metal
clang++ -std=c++20 add.cpp -stdlib=libc++ -I$TT_METAL_HOME -I/$TT_METAL_HOME/tt_metal -I$TT_METAL_HOME/tt_metal/api/ \
    -I$TT_METAL_HOME/tt_metal/third_party/umd/device/api -I$TT_METAL_HOME/tt_metal/hostdevcommon/api \
    -I$TT_METAL_HOME/tt_metal/third_party/tracy/public -I$TT_METAL_HOME/tt_metal/third_party/taskflow/3rd-party \
    -I$TT_METAL_HOME/tt_metal/hw/inc -I$TT_METAL_HOME/ \
    -L$TT_METAL_HOME/build/lib -ltt_metal -ldevice \
    -DFMT_HEADER_ONLY
TT_METAL_DPRINT_CORES=0,0 LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib ./a.out
