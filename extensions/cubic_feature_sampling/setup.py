# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-19 17:03:06
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:02:06
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cubic_feature_sampling',
      version='1.1.0',
      ext_modules=[
          CUDAExtension('cubic_feature_sampling', ['cubic_feature_sampling_cuda.cpp', 'cubic_feature_sampling.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
