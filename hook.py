import torch

# 定义装饰器函数，用来在特定的Tensor对象上触发断言
def assert_on_setitem(original_setitem):
    def new_setitem(self, key, value):
        assert False, "Tensor value cannot be changed"
        original_setitem(self, key, value)
    return new_setitem

# 创建一个原始的 torch.Tensor 对象
x = torch.tensor([1, 2, 3], dtype=torch.float)

# 保存原始的 __setitem__ 方法
original_setitem = x.__setitem__

# 在特定的 Tensor 对象上应用装饰器
x.__setitem__ = assert_on_setitem(original_setitem)

y = 1
# 尝试改变原始的 torch.Tensor 对象的值，会触发断言
try:
    x = y
except AssertionError as e:
    print(e)
