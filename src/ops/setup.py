from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

current_path = os.path.abspath(os.path.dirname(__file__))

op_files = glob.glob("./pytorch/*.cpp") + \
    glob.glob("./kernel/*.cpp") + \
    glob.glob("./kernel/*.cu")

ops_module = CUDAExtension(
    "ops",
    include_dirs=[
        current_path,
        os.path.join(current_path,"include"),
        os.path.join(current_path,"pytorch")
    ],
    sources=op_files
)

setup(
    name="ops",
    ext_modules=[ops_module],
    cmdclass={'build_ext':BuildExtension}
)