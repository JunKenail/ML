"""
Liu Jun
Original Version
dicision tree by C4.5 and ID3
"""
import numpy as np
import turtle as t
class Node:
    def __init__(self, value=None):
        self.value = value
        self.father = None
        self.children = dict()
        self.isleaf = False

    def isaleaf(self, category):
        self.isleaf = True
        self.category = category

class C45:  # C45算法
    def __init__(self):
        self.root = Node()

    def grda(self, data: list, feature: str):  # 计算datadict数据集中feature特征的信息增益比
        findex = self.features.index(feature)
        flist = list(np.array(data)[:, findex])
        if self.h(flist) == 0:
            return 0
        else:
            return self.gda(data, feature)/self.h(flist)

    def h(self, data: list):  # 计算单数据集data的熵
        values = set(data)
        hd = 0
        for v in values:
            hd += -data.count(v)/len(data)*np.log2(data.count(v)/len(data))
        return hd

    def gda(self, data: list, feature: str):  # 计算数据集data中特征feature的信息增益
        dlist = list(np.array(data)[:, -1])
        findex = self.features.index(feature)
        flist = list(np.array(data)[:, findex])
        fset = set(flist)  # 特征feature的值
        gda = self.h(dlist)
        for value in fset:
            dv = []
            for i in range(len(flist)):
                if flist[i] == value:
                    dv.append(dlist[i])
            gda -= self.h(dv)
        return gda

    def bestfeature(self, data: list, features: list):  # 返回数据集data下的特征features中最大信息增益比的特征
        grdas = []
        for f in features:
            grdas.append(self.grda(data, f))
        return features[grdas.index(max(grdas))]

    def creatdicisiontree(self, data: list, features: list):  # 建树
        self.data = data
        self.features = features
        self.datawithfeatures = [self.features]+data
        datalist = [data]
        nodelist = [self.root]
        tempfeatures = features.copy()
        while datalist:
            cdata = datalist.pop(0)
            cdataarray = np.array(cdata)
            ccategory = list(cdataarray[:, -1])
            cnode = nodelist.pop(0)
            if len(set(ccategory)) == 1:
                cnode.isaleaf(list(set(ccategory))[0])
            elif not tempfeatures:
                ck = None
                cn = 0
                for c in set(ccategory):
                    if ccategory.count(c) > cn:
                        ck, cn = c, ccategory.count(c)
                cnode.isaleaf(ck)
            else:
                # print(self.bestfeature(cdata, features))
                cnode.value = self.bestfeature(cdata, tempfeatures)
                findex = self.features.index(cnode.value)
                fdata = list(cdataarray[:, findex])
                fvalue = set(fdata)
                tempfeatures.remove(cnode.value)
                for fv in fvalue:
                    tempd = []
                    tempn = Node()
                    for i in range(len(fdata)):
                        if fdata[i] == fv:
                            tempd.append(cdata[i])
                    datalist.append(tempd)
                    tempn.father = cnode
                    cnode.children[fv] = tempn
                    nodelist.append(tempn)

    def printtree(self):  # 打印树
        # print("The features and data :")
        # for i in self.datawithfeatures:
        #     print(i)
        print("The dicision tree is:")
        nodelist = [self.root]
        while nodelist:
            node = nodelist.pop(0)
            print(str(node.value), end='-')
            if node.children != dict():
                for c in node.children.keys():
                    print(str(c)+":", end='')
                    if node.children[c].isleaf:
                        print(str(node.children[c].category), end='(leaf)')
                        if c != list(node.children.keys())[-1]:
                            print('/', end='')
                    else:
                        print(str(node.children[c].value), end='')
                        if c != list(node.children.keys())[-1]:
                            print('/', end='')
                        nodelist.append(node.children[c])
            print()

    def drawtree(self):  # 画树
        t.pencolor("black")
        t.pensize(2)
        t.hideturtle()
        t.speed(10)
        nl = [[self.root, 0, 200]]
        while nl:
            n = nl.pop(0)
            t.up()
            t.goto(n[1], n[2])
            t.seth(-90)
            t.fd(10)
            t.seth(0)
            t.down()
            t.write(str(n[0].value), font=("楷体", 17, "bold"))
            if n[0].children:
                t.up()
                t.seth(-90)
                t.fd(10)
                t.seth(180)
                t.fd(20)
                t.down()
                cl = list(n[0].children.keys())
                cn = len(cl)
                da = int(180 / (cn + 1))
                dl = int(20*len(str(n[0].value))/(cn + 1))
                for i in range(cn):
                    t.up()
                    t.bk((i + 1) * dl)
                    t.down()
                    t.lt((i+1)*da)
                    t.write(str(cl[i]), font=("楷体", 10, "bold"))
                    t.fd(100)
                    t.up()
                    t.fd(10)
                    if n[0].children[cl[i]].isleaf:
                        t.write(str(n[0].children[cl[i]].category), font=("楷体", 10, "bold"))
                    else:
                        nl.append([n[0].children[cl[i]], t.pos()[0], t.pos()[1]])
                    t.bk(110)
                    t.down()
                    t.seth(180)
                t.seth(0)

class ID3(C45):  # ID3算法
    def bestfeature(self, data: list, features: list):  # 返回数据集data下的特征features中最大信息增益的特征
        gdas = []
        for f in features:
            gdas.append(self.gda(data, f))
        return features[gdas.index(max(gdas))]

def main():
    # 决策树

    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\
~~~~~~~~~~~~~~~~~~~~~~~~~决策树~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    """
    DATA1:
        贷款申请样本数据
        实例样例：[年龄，有工作，有自己的房子，信贷情况，类别]
        年龄：青年-0，中年-1.老年-2
        有工作：否-0，是-1
        有自己的房子：否-0，是-1
        信贷情况：一般-0，好-1，非常好-2
        类别：否-0，是-1

    """
    features1 = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    data1 = [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 0, 1],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 1, 0],
             [1, 1, 1, 1, 1],
             [1, 0, 1, 2, 1],
             [1, 0, 1, 2, 1],
             [2, 0, 1, 2, 1],
             [2, 0, 1, 1, 1],
             [2, 1, 0, 1, 1],
             [2, 1, 0, 2, 1],
             [2, 0, 0, 0, 0]]
    print("----------------DATA1:----------------\nCreate dicision tree by C4.5:")
    c45 = C45()
    c45.creatdicisiontree(data1, features1)
    c45.printtree()
    c45.drawtree()
    Continue = input('''Continue or not?Input "Y" is "Yes" and "N" is "No":''')
    if Continue == "Y":
        t.clear()
    else:
        t.exitonclick()  # 鼠标点击关闭画布，继续运行之后的代码
    print("\nCreate dicision tree by ID3:")
    id3 = ID3()
    id3.creatdicisiontree(data1, features1)
    id3.printtree()
    id3.drawtree()
    Continue = input('''Continue or not?Input "Y" is "Yes" and "N" is "No":''')
    if Continue == "Y":
        t.clear()
    else:
        t.exitonclick()  # 鼠标点击关闭画布，继续运行之后的代码
    """
    DATA2:
        是否购买计算机（ID3data.csv）
    """
    features2 = ["A1", "A2", "A3", "A4"]
    f = open("ID3data.csv", "r")
    data2 = f.readlines()
    data2 = [i.split(',') for i in data2]
    data2 = [i[:5] for i in data2]
    print("\n----------------DATA2:----------------\nCreate dicision tree by C4.5:")
    c45 = C45()
    c45.creatdicisiontree(data2, features2)
    c45.printtree()
    c45.drawtree()
    Continue = input('''Continue or not?Input "Y" is "Yes" and "N" is "No":''')
    if Continue == "Y":
        t.clear()
    else:
        t.exitonclick()  # 鼠标点击关闭画布，继续运行之后的代码
    print("\nCreate dicision tree by ID3:")
    id3 = ID3()
    id3.creatdicisiontree(data2, features2)
    id3.printtree()
    id3.drawtree()
    t.exitonclick()

if __name__ == '__main__':
    main()