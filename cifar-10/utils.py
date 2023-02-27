import os
import torch


def convert_arg_line_to_args(arg_line):
    """
     using for splitting args name and value
     用于切分配置文件的参数
    """
    return arg_line.split()


# using for splitting the str data and put them into a list
# 用于将参数配置中的字符gpu改成整型数组，到时候DP模式指定GPU直接喂
# 用于将学习率组和对应的patience，eval_freq分别塞入列表
def str2list(string, change_type: str):
    """
    Args:
        string: the input string(must split by ,),like 1,2,3,4   0.5,0.2
        change_type: the type you want to change ['int','float','str']

    examples  string : 1,2,3,4    type :int
    Returns: list [1,2,3,4]

    """
    assert change_type in ['int', 'float', 'str'], 'type choose from int ,float or str'
    data_list = []
    str_list = string.split(',')

    if change_type == 'int':
        for elem in str_list:
            data_list.append(int(elem))
    elif change_type == 'float':
        for elem in str_list:
            data_list.append(float(elem))
    elif change_type == 'str':
        data_list = str_list
    return data_list

def check_mkdir(path):
    """
    check the path exist or not ,if not ,create path
    """
    if not os.path.exists(path):
        os.makedirs(path)

