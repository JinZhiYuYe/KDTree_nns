# -*- coding=utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Node:
    """结点数据结构"""

    def __init__(self, position, split, value, left=None, right=None):
        self.position = position  # 切分点位置
        self.split = split  # 切分维度
        self.value = value  # 目标
        self.left = left  # 左结点
        self.right = right  # 右节点

        self.tag = 0  # 是否被标记为最小值


class Stack(object):
    """栈"""

    def __init__(self):
        self.__items = []

    def push(self, item):
        """入栈"""
        self.__items.append(item)

    def pop(self):
        """出栈"""
        return self.__items.pop()

    def peek(self):
        """返回栈顶元素"""
        if self.__items:
            return self.__items[-1]
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return not self.__items

    def size(self):
        """栈中元素个数"""
        return len(self.__items)

    def cout(self):
        print(self.__items)


class KDTree(object):
    def __init__(self, train_x, train_y):
        self.root = self.createKDTree(train_x, train_y, split=0)

    def createKDTree(self, dataset, target, split):
        """
        :param dataset: 数据集
        :param split: 切分维度
        :return:
        """
        if not dataset.shape[0]:
            return None

        sorted_index = np.argsort(dataset[:, split])  # 数据排序

        median = dataset.shape[0] // 2  # 数据中点
        split_position = dataset[sorted_index[median]]  # 切分点
        value = target[sorted_index[median]]

        node = Node(position=split_position, split=split, value=value)
        split = (split + 1) % D_in   # 取模
        node.left = self.createKDTree(dataset[sorted_index[:median]], target[sorted_index[:median]], split)
        node.right = self.createKDTree(dataset[sorted_index[median + 1:]], target[sorted_index[median + 1:]], split)
        return node  # 返回根结点


class Knn(object):
    def __init__(self, tree, k=3, p=2):
        """
        :param k:
        :param p:
        """
        self.k = k
        self.p = p
        self.tree = tree
        self.__nearest_node = None
        self.__search_stack = Stack()
        self.__further_stack = Stack()
        self.k_neighbour = []

    def __distance(self, x, y):
        """计算距离"""
        return np.linalg.norm(x - y, ord=self.p)

    def __fit(self, tree, X):
        for i in range(self.k):
            self.__find(tree.root, X)
        # print(self.__k_neighbour)
        return self.k_neighbour

    def __travel(self, node, X):
        """形成搜索路径"""
        if not node:
            return None
        while node:
            s = node.split  # 切分维度
            posi = node.position
            if posi[s] >= X[s]:
                self.__search_stack.push(node)
                self.__further_stack.push(node.right)
                node = node.left

            else:
                self.__search_stack.push(node)
                self.__further_stack.push(node.left)
                node = node.right

    def __find(self, node, X):
        self.__travel(node, X)  # 初始搜索路径
        self.__nearest_node = self.__search_stack.pop()  # 当前最近点（叶子结点）
        self.__further_stack.pop()  # 去掉叶结点的左（右）空结点
        if self.__nearest_node.tag == 0:  # 可能更近点， 没有被标记为最近点
            mindist = self.__distance(self.__nearest_node.position, X)  # 当前最近距离
            node_visited = 1  # 计算次数
        else:
            mindist = np.inf
            node_visited = 0

        while True:

            if self.__search_stack.is_empty():  # 堆栈空，回溯结束，当前最近点即为最近点
                self.__nearest_node.tag = 1  # 标记为最近点
                self.k_neighbour.append(self.__nearest_node)
                if len(self.k_neighbour) == self.k:
                    # print(self.__k_neighbour)
                    return
                break

            split_node = self.__search_stack.pop()  # 切分点
            further_node = self.__further_stack.pop()  # 切分超平面的左（右）空间

            split = split_node.split
            split_dist = abs(split_node.position[split] - X[split])  # 目标点到分割超平面距离
            if mindist < split_dist:  # 当前最近距离小于到超平面距离，左（右）空间一定没有更近点，回溯到上一个根结点
                continue

            # 当前最近距离大到超平面距离，左（右）空间可能有更近点
            if split_node.tag == 0:  # 可能更近点， 没有被标记为最近点

                temp_dist = self.__distance(split_node.position, X)  # 计算目标点与分割点的欧氏距离
                node_visited += 1

                if temp_dist < mindist:  # 到分割点距离小于当前最近距离， 更新
                    mindist = temp_dist
                    self.__nearest_node = split_node

            if further_node:  # 左（右）空间存在叶子结点

                self.__travel(further_node, X)  # 更新路径
                nearer_node = self.__search_stack.pop()  # 左（右）空间的叶子结点可能为更近点
                self.__further_stack.pop()  # 去掉叶结点的左（右）空结点

                if nearer_node.tag == 0:  # 可能更近点， 没有被标记为最近点

                    temp_dist = self.__distance(nearer_node.position, X)
                    node_visited += 1
                    if temp_dist < mindist:  # 到左（右）空间的叶子结点小于当前最近距离， 更新
                        mindist = temp_dist
                        self.__nearest_node = nearer_node

    def __clear_tag(self, root):
        """清除标记"""
        if root:
            if root.tag == 1:
                root.tag = 0
            self.__clear_tag(root.left)
            self.__clear_tag(root.right)

    def predict(self, X):
        count = {}
        k_neighbors = self.__fit(self.tree, X)
        for node in k_neighbors:
            count[node.value] = count.get(node.value, 0) + 1
        self.__clear_tag(self.tree.root)  # 清除结点标记
        k_neighbors.clear()  # 清除最近点信息
        y_pred = max(count.keys(), key=lambda x: count[x])  # 决策规则：多数表决

        return y_pred

    def valitation(self, dataset, X):
        ls = []
        for i in range(N_t):
            ls.append((self.__distance(dataset[i], X), dataset[i]))

        ls.sort(key=lambda x: x[0])


def load_dataset():
    """加载数据集"""
    np.set_printoptions(suppress=True)
    dataset = np.loadtxt(r"data.csv", delimiter=",", skiprows=1, usecols=np.arange(1, 32), dtype=str)

    # print(datase)
    # df = pd.read_csv(r"./dataset/data.csv").iloc[0:, 1:-1]
    # dataset = df.to_numpy()
    # print(dataset)
    train_y, train_x = np.split(dataset, (1,), axis=1)
    train_y = np.array([0 if i == 'M' else 1 for i in train_y.ravel()])

    train_x, val_x = np.split(train_x, (300, ), axis=0)
    train_y, val_y = np.split(train_y, (300, ), axis=0)
    return train_x.astype("float"), train_y.astype("int"), val_x.astype("float"), val_y.astype("int")


train_x, train_y, val_x, val_y = load_dataset()

# 数据量、维度
N_t, D_in = train_x.shape
print("训练集：", end='')
print(N_t, D_in)
print("验证集：", end='')
N_v, D_in = val_x.shape
print(N_v, D_in)

# print("正在构造kd树。。。")
# 构造kd树
kdtree = KDTree(train_x, train_y)
# print("kd树构造完成")
print("\n")

acc_ls = []



# p_can = [1, 2, np.inf]
# for p in p_can:
#     knn = Knn(tree=kdtree, k=3, p=p)
#     pred_ls = []
#     for i in range(N_v):
#         y_pred = knn.predict(val_x[i])
#         pred_ls.append(y_pred)
#     acc = sum(pred_ls==val_y)/N_v
#     acc_ls.append(acc)
#     print("\n")
#     print("p值：", p)
#     print("准确率：", acc)
#
# p_can = ['1', '2', 'np.inf']
# plt.xticks(range(0, 3), p_can)
# plt.plot(p_can, acc_ls, marker='o')
# plt.xlabel("P_can")
# plt.ylabel("accuracy rating")
# plt.show()





# k_can = range(1, 21)
# for k in k_can:
#     knn = Knn(tree=kdtree, k=k, p=2)
#     pred_ls = []
#     for i in range(N_v):
#         y_pred = knn.predict(val_x[i])
#         pred_ls.append(y_pred)
#     acc = sum(pred_ls==val_y)/N_v
#     acc_ls.append(acc)
#     print("\n")
#     print("k值：", k)
#     print("准确率：", acc)
#
#
# plt.plot(k_can, acc_ls, marker='o')
# plt.xticks(range(1, 21))
# plt.xlabel("K_can")
# plt.ylabel("accuracy rating")
# plt.show()


np.random.seed(0)
bagging_ls = []
for j in range(1, 21):
    acc = []
    per_model = []
    for i in range(j):
        index_ls = np.random.randint(low=0, high=300, size=(20, 1)).ravel()
        feature_ls = np.random.randint(low=0, high=30, size=(2, 1)).ravel()
        print(feature_ls)
        D_in = 2
        rd_train_x = train_x[index_ls][:, feature_ls]
        rd_train_y = train_y[index_ls]

        # print(rd_train_y)

        # print("正在构造kd树。。。")
        # 构造kd树
        kdtree = KDTree(rd_train_x, rd_train_y)
        # print("kd树构造完成")
        acc = []
        # k_can = [i for i in range(1, 16)]
        # for k in k_can:

        knn = Knn(tree=kdtree, k=7, p=2)
        # print(k, 2)
        # print(feature)
        pred_ls = []
        for i in range(N_v):
            y_pred = knn.predict(val_x[i, feature_ls])
            pred_ls.append(y_pred)

        per_model.append(pred_ls)

    per_model = np.array(per_model)

    bagging_pred = []
    for i in range(N_v):
        count = {}
        for i in per_model[:, i]:
            count[i] = count.get(i, 0) + 1
        y_pred = max(count.keys(), key=lambda x: count[x])  # 决策规则：多数表决
        bagging_pred.append(y_pred)

    print("弱分类器个数：", j)
    print("准确率：", sum(bagging_pred==val_y)/N_v)
    print("\n")
    bagging_ls.append([j, sum(bagging_pred==val_y)/N_v])


bagging_ls = np.array(bagging_ls)
plt.plot(bagging_ls[:, 0], bagging_ls[:, 1], marker='o')
plt.xlabel("model number")
plt.ylabel("accuracy rating")
plt.xticks(range(1, 21))
plt.show()


# np.random.seed(2)
# feature_ls = np.random.randint(0, 30, size=(30, 2))
# for feature_index in feature_ls:
#     x = train_x[:, feature_index]
#
#     plt.scatter(x[:, 0], x[:, 1], c=train_y)
#     plt.show()
