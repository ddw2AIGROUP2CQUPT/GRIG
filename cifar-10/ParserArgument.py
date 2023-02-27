# -*- coding: utf-8 -*-
# @Time : 2022/10/21 2:04 PM
# @Author : Li Zhany
# @Email : 949777411@qq.com
# @File : ParserArgument.py
# @
# @Project : GAT_base


import argparse
import sys
from utils import convert_arg_line_to_args

parser = argparse.ArgumentParser(description="GAT_base", fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help="the model name", default='test_model')
parser.add_argument('--gpu', type=str, help='gpu id',default='0,1,2,3,4,5,6')

# dataset

parser.add_argument('--data_path', type=str, help="the path of your train datasets", default='/home/ubuntu/lxd-workplace/yl/yl/cifar-10/data_img2graph_v1/train')

parser.add_argument('--eval_data_path', type=str, help="the path of your datasets using for online eval", default='/home/ubuntu/lxd-workplace/yl/yl/cifar-10/data_img2graph_v1/val')
parser.add_argument('--img_size',       type=int, help='the raw image size you use for generate graph',  default='32')

# training
parser.add_argument('--batch_size', type=int, help="batch size", default=64)
parser.add_argument('--total_steps', type=int, help='the total iteration number', default=100000)
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--lr_group', type=str, help='the learning rate group', default='0.4,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001')
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-4)

# log and save
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')

parser.add_argument('--log_directory', type=str, help='directory to save summaries', default='./modelInfos/')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=100)
# online eval
parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',
                    action='store_true')
parser.add_argument('--eval_freq_group', type=str, help='Online evaluation frequency in global steps',
                    default='200,100,50,25,20,10')
parser.add_argument('--eval_summary_directory', type=str, help='output directory for eval summary,'
                                                               'if empty outputs to checkpoint folder', default='')
parser.add_argument('--patience_group', type=str, help='patience times to adjust lr if eval acc can not be better',
                    default="10,20,40,80,100,200")

if sys.argv.__len__() == 2:
    args_file_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([args_file_with_prefix])

else:
    args = parser.parse_args()

# vars是返回对象object的属性和属性值的字典对象，这样就可以 check argument elegantly了
print("*"*20,'Args', "*"*20)
for arg in vars(args):
    # arg是键，用 getattr 获取args中键为arg对应的值
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

print("*"*50)
