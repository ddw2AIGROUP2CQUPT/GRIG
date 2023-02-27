from Dataset.Base_dataset import *


class DatasetBaseClassification(DatasetBase):

    def __init__(self,classes, mode, args, ):
        """
           用于分类任务的数据集类
               :param mode: choose from train or val ,it only related to the data path
               :param args: config
           """
        self.CLASSES = classes  # 具体数据集导入具体的类字典

        assert mode in ['train', 'val'], 'mode must choose from train or val！'
        self.mode = mode
        if self.mode == 'train':
            self.data_path = args.data_path  # 训练集路径
        elif self.mode == 'val':
            self.data_path = args.eval_data_path  # 验证集路径

        # 获取数据的路径
        self.graph_path_list, self.class_id_list = self.__getdataInfos__()

        self.normalize = None

    def __getdataInfos__(self):
        graph_path_list = []
        graph_class_list = []
        classnames = os.listdir(self.data_path)
        for classname in classnames:
            dir_path = os.path.join(self.data_path, classname)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                cls = self.CLASSES[classname]
                graph_path_list.append(file_path)
                graph_class_list.append(cls)
        return graph_path_list, graph_class_list

    def __getitem__(self, idx):

        graph_path = self.graph_path_list[idx]  # 样本x的路径
        class_id = self.class_id_list[idx]  # 标签y的路径

        f = h5py.File(graph_path, 'r')
        nodes = torch.as_tensor(np.array(f['x']), dtype=torch.float)  # 结点特征
        adj = torch.as_tensor(np.array(f['adj']), dtype=torch.long)  # 邻接矩阵
        edge_attr = torch.as_tensor(np.array(f['edge_attr']), dtype=torch.float)  # 边特征
        y = torch.as_tensor([class_id], dtype=torch.int)  # 标签

        if self.normalize is not None:
            nodes = self.normalize(nodes)

        sample = Data(x=nodes, edge_index=adj, edge_attr=edge_attr, y=y)
        return sample

    # 返回类别数
    def get_class_dim(self):
        return len(self.CLASSES)

    # 返回结点特征维度
    def get_dim_node_features(self):
        sample = self.__getitem__(0)
        nodes = sample.x
        return nodes.shape[1]

    def __len__(self):
        return len(self.graph_path_list)


class Cifar10Dataset(DatasetBaseClassification):
    classes = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7,
               "ship": 8, "truck": 9}

    def __init__(self, mode, args):
        """
        Cifar10 Classification Dataset
        :param mode:
        :param args:
        """
        super(Cifar10Dataset, self).__init__(Cifar10Dataset.classes, mode, args, )

