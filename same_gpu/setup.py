from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_p2p_kenan',
    ext_modules=[
        CUDAExtension(
            name='cuda_p2p_kenan',
            sources=['cuda_p2p_kenan.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)