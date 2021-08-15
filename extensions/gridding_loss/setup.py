# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-30 11:03:55
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-30 11:13:39
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='gridding_distance',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('gridding_distance', ['gridding_distance_cuda.cpp', 'gridding_distance.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
