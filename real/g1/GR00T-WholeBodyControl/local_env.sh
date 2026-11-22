#!/usr/bin/env bash
# Source from bash or zsh. Dependencies stay inside this checkout.
if [ -n "${ZSH_VERSION:-}" ]; then
    SONIC_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
    SONIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export SONIC_ROOT
export TensorRT_ROOT="$SONIC_ROOT/.local/TensorRT-10.13.3.9"
export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT="$CUDA_HOME"
export onnxruntime_ROOT="$SONIC_ROOT/.local/onnxruntime"
export HAS_ROS2=0
export CMAKE_PREFIX_PATH="$SONIC_ROOT/.local/deps/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$SONIC_ROOT/.local/deps/usr/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$SONIC_ROOT/.local/deps/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:$CUDA_HOME/lib64:$SONIC_ROOT/.local/deps/usr/lib/x86_64-linux-gnu:$SONIC_ROOT/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
