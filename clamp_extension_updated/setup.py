from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='clamp_custom_cuda',
      ext_modules=[
          CUDAExtension(name='clamp_custom_cuda', sources=[
              'clamp_custom_cuda.cpp',
              'clamp_custom_cuda_kernel.cu',
              ], extra_compile_args={'cxx': ['-g', '-O3'],
                                     'nvcc': ['-g', '-O3', '-std=c++17']})
      ],
      cmdclass={
          'build_ext' : BuildExtension
      })
