为了运行这个项目首先需要
1. 进入到src/ops/ 文件夹里面 运行python setup.py install 注意这个需要每次更新算子都需要运行一次！！！
2. 然后将uniad里面的/data文件夹复制到Experiment-UniAD/src/ 下面（应该是需要使用里面的内容的 具体在哪里有点忘了）目录的结构大概是：
```
├── main.py
├── readme.md
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── data
│   ├── mmdet3d
│   ├── modules
│   ├── motion_head
│   ├── occ_head
│   ├── ops
│   ├── seg_head
│   ├── track_head
│   ├── uniad_e2e.py
│   ├── uniad_track.py
│   └── utils
└── tests
    ├── __init__.py
    ├── __pycache__
    ├── inputs
    ├── motion_head_torch
    ├── occ_head_torch
    ├── test_builder.py
    ├── test_module.py
    └── weights
```
3. 每次提交前在tests文件夹下面运行pytest test_modules.py 和 pytest test_builder.py