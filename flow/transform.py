from .tensor import Tensor
import functools


def Normalize(std, mean):
    def f(tensor):
        return (tensor - mean) / std
    return f

# def ToTensor():
#     @functools.wraps
#     def f(data):
#         return Tensor(data)
#     return f

def Compose(*func_list):
    def f(data):
        for func in func_list:
            data = func(data)
        return data
    return f