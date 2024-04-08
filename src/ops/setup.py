from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_path = os.path.abspath(os.path.dirname(__file__))
ops_module = CUDAExtension(
    "ops",
    include_dirs=[
        current_path,
        os.path.join(current_path,"include"),
        os.path.join(current_path,"pytorch")
    ],
    sources=[
        'kernel/ms_deform_attn_cuda.cu',
        'pytorch/ms_deform_attn.cpp'
    ]
)

setup(
    name="ops",
    ext_modules=[ops_module],
    cmdclass={'build_ext':BuildExtension}
)