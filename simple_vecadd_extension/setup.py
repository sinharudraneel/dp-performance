from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='vector_add',
      ext_modules=[
          CUDAExtension('vector_add__cuda', [
              'vector_add_cuda.cpp',
              'vector_add_cuda_kernel.cu',
          ])
      ],
      cmdclass={
          'build_ext' : BuildExtension
      })
