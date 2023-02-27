import time

from torch.utils.tensorboard import SummaryWriter

from ParserArgument import args
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import torch
from tqdm import tqdm
import numpy as np
import torch_geometric
from torch_geometric.loader import DataListLoader
from utils import check_mkdir, str2list

# 此处名称和模型层级的架构方式待优化
from model import GCN_8_plus, GCN_Layer_4, GCN


# 计算训练集的均值和方差
def calculate_mean_std(dataset, img_size):
    """

    :param dataset: 训练集
    :param img_size:  用于 将计算得到的mean和std
    前两维度（聚类中心位置）替换成图像大小的一半
    :return: mean and std after replace
    """

    # 训练数据集
    train_dataset = dataset

    trainloader = DataListLoader(train_dataset, batch_size=1, num_workers=0,
                                 shuffle=False)
    length = len(train_dataset)
    std_sum, feats_sum, num_batches = 0, 0, 0
    for data in tqdm(trainloader):
        x = data[0].x
        mean_sample = torch.mean(x, dim=0, keepdim=True)
        feats_sum += mean_sample
        sqrt_sample = torch.mean(x ** 2, dim=0, keepdim=True)

        std_sample = (sqrt_sample - mean_sample ** 2) ** 0.5
        if np.isnan(std_sample).any():
            print(x[:, 9])
            print(std_sample)
            print('sqrt', sqrt_sample)
            print('mean', mean_sample)
            print('mean suqare', mean_sample ** 2)
            print(sqrt_sample - mean_sample ** 2)
            return
        std_sum += std_sample
    # print('MEAN SUM',feats_sum)
    # print("std sum",std_sum)
    mean = feats_sum / length  # 均值
    std = std_sum / length  # 标准差
    torch.set_printoptions(threshold=torch.inf)

    mean[0][:2] = img_size / 2
    std[0][:2] = img_size / 2
    print('*' * 40)
    print("mean:", mean)
    print("std", std)
    print("*" * 40)
    return mean, std


# 模型学习率调整类
class MyScheduler:
    def __init__(self, lr_groups: list, ):

        self.lr_groups = lr_groups

    def update_lr(self, optimizer):
        for i, param_group in enumerate(optimizer.param_groups):
            if len(self.lr_groups) != 1:
                param_group['lr'] = self.lr_groups.pop(0)
            else:
                param_group['lr'] = self.lr_groups[0]


# 在线验证
def online_eval(model, dataloader_eval, criterion):
    correct = 0
    test_loss = 0.0
    test_total = 0
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval)):
        with torch.no_grad():
            labels_eval = [sb.y for sb in eval_sample_batched]
            labels_eval = torch.cat(labels_eval).cuda()

            outputs = model(eval_sample_batched)
            loss = criterion(outputs, labels_eval.long())

            test_loss += loss.item()
            test_total += labels_eval.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels_eval).sum().cuda()

    eval_loss = test_loss / test_total
    eval_acc = 100 * correct / test_total
    return eval_loss, eval_acc


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def train(args):

    random_state = 1
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)

    # ==== 创建 保存路径 的文件夹 ====
    if args.model_name == "":  # 不想取名字那就帮你取
        # 此处模型名字也可自己DIY
        args.model_name = "GCN_bs-{}_lr-{}_wd-{}".format(args.batch_size, args.lr_group, args.weight_decay)
    model_info_savepath = os.path.join(args.log_directory, args.model_name)
    checkpoint_rootpath = os.path.join(model_info_savepath, 'cp')

    check_mkdir(model_info_savepath)
    check_mkdir(checkpoint_rootpath)

    command = 'cp ' + sys.argv[0] + ' ' + model_info_savepath
    if len(sys.argv) > 1:
        os.system(command)
        command = 'cp ' + sys.argv[1] + ' ' + model_info_savepath
    os.system(command)
    command = 'rm -rf ' + checkpoint_rootpath + '/*'
    os.system(command)

    torch.cuda.empty_cache()
    use_gpu = torch.cuda.is_available()
    device_ids = list(range(torch.cuda.device_count()))
    gpu_group = str2list(args.gpu, change_type='int')
    print("gpu group",gpu_group)
    print("*" * 15, "\nif cuda available:{}".format(use_gpu))
    print("devices ids :",device_ids)
    print("the gpu will be used:", gpu_group)
    # ==== init some vars ====
    global_step = 0  # 当前的步数
    total_steps = args.total_steps  # 总步数
    best_eval_acc = 0  # 最好的准确度
    best_eval_steps = 0  # 最好准确度对应的步数
    patience = 0
    criterion = torch.nn.CrossEntropyLoss()

    lr_group = str2list(args.lr_group, change_type='float')  # 这是存学习率的列表
    eval_freq_group = str2list(args.eval_freq_group, change_type='int')  # 这是存验证频率的列表
    patience_group = str2list(args.patience_group, change_type='int')  # 存最大等待的eval次数的列表，指标没上去就下调lr

    # ==== about dataset ====
    from Data_loader.ClassificationBase_dataset import Cifar10Dataset
    dataset = Cifar10Dataset(mode='train', args=args)  # train set
    val_dataset = Cifar10Dataset(mode='val', args=args)

    # 自动计算归一化的数据
    if (dataset.normalize or val_dataset.normalize) is None:
        mean, std = calculate_mean_std(dataset, args.img_size)
        from Data_loader.transforms import Normalize
        normalize = Normalize(mean, std)
        dataset.normalize = normalize
        val_dataset.normalize = normalize
    dataloader = DataListLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=True)
    dataloader_eval = DataListLoader(val_dataset, batch_size=args.batch_size)
    print("train data number:{},eval data number:{}".format(len(dataset), len(val_dataset)))

    # ==== about model ====
    # 此处模型自己DIY

    num_features = dataset.get_dim_node_features()  # 获取节点维度数
    num_classes = dataset.get_class_dim()  # 获取类别数
    
    model = GCN_8_plus(num_features=num_features, num_classes=num_classes, initdim=16, inithead=16)
    # model = GCN_Layer_4(num_features=num_features, num_classes=num_classes, initdim=16, inithead=24)
    # model = GCN(num_features = num_features, num_classes = num_classes, init_out_dim = 16, init_head_num = 16)

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("== Total number of learning parameters: {}".format(num_params_update)) 
    model.train()
    model = torch_geometric.nn.DataParallel(model, device_ids=gpu_group)
    model = model.cuda()
    print("== Model Initialized")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_group.pop(0), weight_decay=args.weight_decay)
    scheduler = MyScheduler(lr_groups=lr_group)
    # ====新建 tensorboard 日志文件 ====
    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
    if args.do_online_eval:
        if args.eval_summary_directory != '':
            eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
        else:
            eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
        eval_summary_writter = SummaryWriter(eval_summary_path, flush_secs=30)

    start_time = time.time()
    duration = 0

    index_eval_freq = 0
    current_eval_freq = eval_freq_group[index_eval_freq]
    current_patience = patience_group[index_eval_freq]  # 当前学习率下的等待耐心，即current_eval_freq*current_patience个连续step指标不提升，下调lr）
    
    while global_step < total_steps:

        print('*' * 10, 'start training')

        # 超过等待耐心，调整lr,eval_freq,patience
        if patience >= current_patience:
            patience = 0
            index_eval_freq += 1

            scheduler.update_lr(optimizer=optimizer)

            if len(eval_freq_group) <= index_eval_freq:
                index_eval_freq = len(eval_freq_group) - 1
                
            current_eval_freq = eval_freq_group[index_eval_freq]
            current_patience = patience_group[index_eval_freq]

        for step, sample_batched in enumerate(dataloader):
            optimizer.zero_grad()

            before_op_time = time.time()

            train_labels = [sb.y for sb in sample_batched]
            train_labels = torch.cat(train_labels).cuda()

            outputs = model(sample_batched)

            loss = criterion(outputs, train_labels.long())
            loss.backward()

            pred = outputs.argmax(dim=1)

            train_correct = (pred == train_labels).sum().cuda()
            train_acc = train_correct / train_labels.size(0)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()

            print(
                '[global step/total steps]: [{}/{}], lr: {:.6f}, loss: {:.8f}, train acc：{:.8f}'.format(global_step,
                                                                                                       total_steps,
                                                                                                       current_lr,
                                                                                                       loss,
                                                                                                       train_acc))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1
            duration += time.time() - before_op_time

            if global_step and global_step % args.log_freq == 0:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (total_steps / global_step - 1.0) * time_sofar

                print("{}".format(args.model_name))
                print_string = ' examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(examples_per_sec, loss, time_sofar, training_time_left))

                writer.add_scalar('train_loss', loss, global_step)
                writer.add_scalar('train_acc', train_acc, global_step)

                writer.add_scalar('lr', current_lr, global_step)
                writer.flush()

            if args.do_online_eval and global_step % current_eval_freq == 0:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_loss, eval_acc = online_eval(model, dataloader_eval, criterion)
                    eval_summary_writter.add_scalar('eval_loss', eval_loss, global_step)
                    eval_summary_writter.add_scalar('eval_accuracy', eval_acc, global_step)

                    is_best = False

                    if eval_acc > best_eval_acc:
                        old_best = best_eval_acc
                        best_eval_acc = eval_acc
                        is_best = True

                    if is_best:
                        patience = 0
                        old_best_step = best_eval_steps
                        old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, 'eval_acc',
                                                                          old_best)
                        model_path = checkpoint_rootpath + old_best_name
                        if os.path.exists(model_path):
                            command = 'rm {}'.format(model_path)
                            os.system(command)
                        best_eval_steps = global_step
                        model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, 'eval_acc',
                                                                            eval_acc)
                        print('New best for {}. Saving model: {}'.format('eval_acc', model_save_name))
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_eval_acc': best_eval_acc,
                                      'best_eval_steps': best_eval_steps
                                      }
                        torch.save(checkpoint, checkpoint_rootpath + model_save_name)
                    else:
                        patience += 1

                eval_summary_writter.flush()

            model.train()
            block_print()
            enable_print()

            global_step += 1

    print("train finished!")


if __name__=='__main__':
    train(args)
