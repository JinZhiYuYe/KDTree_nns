# -*- coding=utf-8 -*-

import numpy as np
import time


def load_dataset():
    # train_x = np.array([[2, 3],
    #                     [5, 4],
    #                     [9, 6],
    #                     [4, 7],
    #                     [8, 1],
    #                     [7, 2]])
    np.random.seed(2)
    train_x = np.random.random(size=(300000, 3))

    return train_x


class Node:
    """结点数据结构"""
    def __init__(self, position, split, left=None, right=None):
        self.position = position  # 切分点位置
        self.split = split  # 切分维度
        # self.value = value  # 目标
        self.left = left   # 左结点
        self.right = right  # 右节点


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
    def __init__(self, dataset):
        self.root = self.createKDTree(dataset, split=0)

    def createKDTree(self, dataset, split):
        """
        :param dataset: 数据集
        :param split: 切分维度
        :return:
        """
        if not dataset.shape[0]:
            return None
        sorted_index = np.argsort(dataset[:, split])          # 数据排序
        median = dataset.shape[0] // 2                  # 数据中点
        split_position = dataset[sorted_index[median]]                # 切分点
        node = Node(position=split_position, split=split)
        split = (split + 1) % D_in                      # 下次切分维度

        node.left = self.createKDTree(dataset[sorted_index[:median]], split)
        node.right = self.createKDTree(dataset[sorted_index[median + 1:]], split)
        return node    # 返回根结点


class Knn(object):
    def __init__(self, k=3, p=2):
        """
        :param k:
        :param p:
        """
        self.k = k
        self.p = p
        self.nearest_node = None
        self.search_stack = Stack()
        self.further_stack = Stack()

    def distance(self, x, y):
        """计算距离"""
        return np.linalg.norm(x-y, ord=self.p)

    def fit(self, tree, X):
        return self.find(tree.root, X)

    def travel(self, node, X):
        """形成搜索路径"""
        if not node:
            return None
        while node:
            s = node.split  # 切分维度
            posi = node.position
            if posi[s] >= X[s]:
                self.search_stack.push(node)
                self.further_stack.push(node.right)
                node = node.left

            else:
                self.search_stack.push(node)
                self.further_stack.push(node.left)
                node = node.right

    def find(self, node, X):
        self.travel(node, X)  # 初始搜索路径
        self.nearest_node = self.search_stack.pop()      # 当前最近点（叶子结点）
        self.further_stack.pop()   # 去掉叶r结点的左（右）空结点
        mindist = self.distance(self.nearest_node.position, X)   # 当前最近距离
        node_visited = 1    # 计算次数

        while True:
            if self.search_stack.is_empty():    # 堆栈空，回溯结束，当前最近点即为最近点
                return [self.nearest_node.position, mindist, node_visited]

            split_node = self.search_stack.pop()   # 切分点
            further_node = self.further_stack.pop()     # 切分超平面的左（右）空间

            split = split_node.split
            split_dist = abs(split_node.position[split] - X[split])  # 目标点到分割超平面距离
            if mindist < split_dist:  # 当前最近距离小于到超平面距离，左（右）空间一定没有更近点，回溯到上一个根结点
                continue

            # 当前最近距离大到超平面距离，左（右）空间可能有更近点
            temp_dist = self.distance(split_node.position, X)  # 计算目标点与分割点的欧氏距离
            node_visited += 1

            if temp_dist < mindist:  # 到分割点距离小于当前最近距离， 更新
                mindist = temp_dist
                self.nearest_node = split_node

            if further_node:   # 左（右）空间存在叶子结点
                self.travel(further_node, X)  # 更新路径
                nearer_node = self.search_stack.pop()    # 左（右）空间的叶子结点可能为更近点
                self.further_stack.pop()     # 去掉叶结点的左（右）空结点
                temp_dist = self.distance(nearer_node.position, X)
                node_visited += 1

                if temp_dist < mindist:  # 到左（右）空间的叶子结点小于当前最近距离， 更新
                    mindist = temp_dist
                    self.nearest_node = nearer_node

    def valitation(self, dataset, X):
        ls = []
        for i in range(N):
            ls.append((self.distance(dataset[i], X), dataset[i]))
        ls.sort(key=lambda x: x[0])
        return list(reversed(min(ls)))+[N]


train_x = load_dataset()

N, D_in = train_x.shape

print("数据维度:", train_x.shape)

knn = Knn(k=1, p=2)
# np.random.seed(1)
val_x = np.random.random(size=(10, 3))


# kdtree = KDTree(train_x)
# print(knn.fit(kdtree, np.array([0.1, 0.2, 0.3])))
# print(knn.valitation(train_x, np.array([0.1, 0.2, 0.3])))


print("\n")
a = time.time()
kdtree = KDTree(train_x)

b = time.time()
print("kd树构造用时：%f" % (b-a))
c = time.time()

for x in val_x:
    s1 = knn.fit(kdtree, x)
    print(f"{x}的最近点为：{s1[0]}，距离：{s1[1]}，运算次数：{s1[2]}")
d = time.time()
print("利用kd树找最近点用时：%f" % (d-c))
print("kd树用时：", (b-a)+(d-c))


print("\n")

c = time.time()
for x in val_x:
    s2 = knn.valitation(train_x, x)
    print(f"{x}的最近点为：{s2[0]},距离：{s2[1]}，运算次数：{s2[2]}")
d = time.time()
print("传统方法用时：%f" % (d-c))



