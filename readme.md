注意：我在现在的仓库里面没有维护inputs和weights了 因为他们属于lfs 有容量上限 另外我没有保存关于forward_test连续inputs的处理 要自行编写代码处理

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
4. 注意！！！ 如果提交了大的文件超过100m 需要先git lfs track <file> 然后再提交

如何验证正确性：

1. 在UniAD原仓库前面用 torch.save(self.seg_head.state_dict(),"/tmp/seg_head_weights.pth") 这样来保存weights，同时需要保存输入、利用pickle保存（这里GPT一下如何保存模型输入就行）
2. 在当前项目的model上load weights 如何把inputs拿出来对进去跑就行了，这样就不用管之前的网络结果是不是正确的
