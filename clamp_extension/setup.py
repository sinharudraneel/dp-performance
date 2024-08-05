from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='clamp_custom_cuda',
      ext_modules=[
          CUDAExtension('clamp_custom_cuda', [
              'clamp_custom_cuda.cpp',
              'clamp_custom_cuda_kernel.cu',
          ])
      ],
      cmdclass={
          'build_ext' : BuildExtension
      })
