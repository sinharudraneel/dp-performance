Second set of tests. Two profilers have been used, from https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html 
One to analyse the time consumption of each kernel, and one to visualise it on the chrome viewer. 

Conditions: Single Layer NN, dimensions: 50 input size, 3 output size. One epoch. 

Somehow lesser GPU engagement. model_inference is not duplicated in the trace unlike the first test. gemmSN kernel used instead of the three long named kernels seen in the last test. 
No immediately apparent difference that could indicate anything useful. Try running clip_and_accumulate and add_noise through the profiler to see which kernels are launched and whether any of those kernels show up in the profile for simpletorch/simpleopacus.

Maybe simpletorch/simpleopacus are too simple and the training just does not include the clip/aggregate/noise step. Make model more complex, make sure there is ample time spent in stochastic gradient descent. Limit to one epoch to avoid repetition of trace.
