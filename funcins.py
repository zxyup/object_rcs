import inspect
from typing import Callable, Type
from pprint import pprint

#
# def set_fields_with_params(m: Callable):
#     # if not inspect.ismethod(m):
#     #     raise TypeError("param `m` must be method type.")
#     # args = inspect.getargs(m.__code__)
#     args = inspect.getargspec()
#     print(args)
#     def inner(obj, *args, **kwargs):
#         ...
#     return inner

def set_fields_with_params():
    """
    该函数用于自动将字段值设置
    """
    # 得到调用该函数的方法的帧位置
    caller_frame = inspect.getouterframes(inspect.currentframe())[1]
    # 读取字段名称
    field_names = caller_frame.frame.f_code.co_varnames[:caller_frame.frame.f_code.co_argcount]
    # 获得所有变量值
    all_arg_dict = inspect.getargvalues(caller_frame.frame).locals
    # 筛选出对应字段设置字段值
    obj = all_arg_dict[field_names[0]]
    for field in field_names[1:]:
        setattr(obj, field, all_arg_dict[field])
    

if __name__ == '__main__':
    ...
    # print(a.a)