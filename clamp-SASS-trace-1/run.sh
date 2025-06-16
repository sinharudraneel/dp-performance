set -e

export DYNAMIC_KERNEL_RANGE=""

export CUDA_VERSION="12.8"; export CUDA_VISIBLE_DEVICES="0" ; 
rm -f traces/*
export TRACES_FOLDER=/root/accel-sim-framework/hw_run/traces/device-0/12.8/clamp_test/NO_ARGS; CUDA_INJECTION64_PATH=/root/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so  LD_PRELOAD=/root/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so /root/accel-sim-framework/gpu-app-collection/src/..//bin/12.8/release/clamp_test  ; /root/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing /root/accel-sim-framework/hw_run/traces/device-0/12.8/clamp_test/NO_ARGS/traces ; rm -f /root/accel-sim-framework/hw_run/traces/device-0/12.8/clamp_test/NO_ARGS/traces/*.trace ; rm -f /root/accel-sim-framework/hw_run/traces/device-0/12.8/clamp_test/NO_ARGS/traces/kernelslist 