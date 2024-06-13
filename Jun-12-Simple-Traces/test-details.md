First set of tests. Two profilers have been used, from https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html 
One to analyse the time consumption of each kernel, and one to visualise it on the chrome viewer. 
Conditions: Single Layer NN, dimensions: 3 input size, 1 output size. One epoch. 

No special kernels discernable apart from detach kernel. Nothing that indicates clipping or noise addition. Try bigger input/output sizes. Maybe try more complex models. 
