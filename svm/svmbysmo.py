"""
Liu Jun
Original Version
SVM & multiSVM by SMO
"""
# 注：这里用的proceed.cleveland.data的数据集有五类，即为多分类问题。这里我的想法是：将其转为了对五个特征的五个二分类问题，
# 得到五个类别的五个二分类支持向量机，之后将每个数据点都代入到这五个支持向量机中得到相应的分类值，如[1,-1,-1,-1,-1],并以此
# 作为每个数据点新的特征，用最小二乘回归对五个支持向量机进行加权组合并直接对应类别数{0,1,2,3,4}，得到的预测值以最接近的类别
# 数为类别得到最终的基于各特征二分类支持向量机的多分类回归线行组合支持向量机。

import numpy as np
import random
import matplotlib.pyplot as plt

# 二分类SVM类
class SVMbySMO():
    def __init__(self):
        self.DataSet = []          # 数据集
        self.n = 0                 # 数据集的特征数
        self.C = 0                 # 惩罚因子
        self.tol = []              # 容错率
        self.kValue = {}           # 核函数种类：linear-线性核函数，Gaussian-高斯核函数（这里用高斯核函数）
        self.maxIter = 100         # 最大迭代次数

    # 加载数据集，并将数据储存为列表结构。
    def loadDataSet(self, filename):
        fr = open(filename)
        for line in fr.readlines():
            if '?' in line:   # 这里采取直接将缺省值（异常值）删去处理。
                continue
            lineArr = line.replace('\n', '').strip().split(',')
            for i in range(len(lineArr)):
                lineArr[i] = float(lineArr[i])
            self.DataSet.append(lineArr)
        self.originalDataSet = self.DataSet.copy()

    # 数据集(异常值（缺损值）等)处理及标准化
    def processDataSet(self):  # n为数据集的特征数
        self.X = []                               # 输入的数据集（标准化的）
        self.labelMat = []                        # 储存类别标签
        self.DataSet = np.array(self.DataSet)
        self.labelMat = list(self.DataSet[:, self.n])
        # 数据标准化
        self.X = [list(i) for i in (self.DataSet[:, 0:self.n]-self.DataSet[:, 0:self.n].mean())/self.DataSet[:, 0:self.n].std()]
        self.DataSet = [list(i) for i in self.DataSet]
        self.initparam()

    # 用于寻找数据集中异常值（缺损值）的索引
    # def findindex(self, str):
    #     indx = []
    #     for i in range(len(self.DataSet)):
    #         for j in [index for (index, value) in enumerate(self.DataSet[i]) if value == '?']:
    #             indx.append([i, j])
    #     return indx

    # 通过数据集初始化参数
    def initparam(self):
        self.X = np.mat(self.X)
        self.labelMat = np.mat(self.labelMat).T
        self.m = self.X.shape[0]
        self.lambdas = np.mat(np.zeros((self.m, 1)))          # 拉格朗日乘子向量
        self.eCache = np.mat(np.zeros((self.m, 2)))           # 误差缓存
        self.K = np.mat(np.zeros((self.m, self.m)))           # 储存用于核函数计算的向量
        for i in range(self.m):
            self.K[:, i] = self.kernels(self.X, self.X[i, :])  # kValue

    # 核函数
    def kernels(self, dataMat, A):
        m, n = dataMat.shape
        K = np.mat(np.zeros((m, 1)))
        if list(self.kValue.keys())[0] == 'linear':
            K = dataMat * A.T  # 线性核
        elif list(self.kValue.keys())[0] == 'Gaussian':
            for j in range(m):
                deltaRow = dataMat[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K/(-1*self.kValue['Gaussian']**2))
        else:
            raise NameError('无法识别和函数')
        return K

    # 选择lambda2,从缓存中寻找符合KKT条件并具有最大误差的j
    def chooseJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]      # 更新误差缓存
        # 筛选出符合KKT条件的lambdas
        validEcacheList = self.eCache[:, 0].A.nonzero()[0]
        if len(validEcacheList) > 1:  # 找到误差最大的j
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE =deltaE
                    Ej =Ek
            return maxK, Ej
        else:
            j = self.randJ(i)
            Ej = self.calcEk(j)
        return j, Ej

    # 随机选择一个不等于i的j
    def randJ(self, i):
        j = i
        while j == i:
            j = int(np.random.uniform(0, self.m))
        return j

    # 计算类别误差
    def calcEk(self, k):
        return float(np.multiply(self.lambdas, self.labelMat).T*self.K[:, k]+self.b)-float(self.labelMat[k])

    # 裁剪lambda2
    def clipLambda(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    # SVMbySOM主函数外循环：train
    def train(self):
        self.svIndex = []              # 支持向量下标
        self.sptVects = []             # 支持向量
        step = 0                       # 外循环迭代器
        entireflag = True              # 扫描标志位
        lambdaPairsChanged = 0
        # 终止条件：超过最大迭代次数，或未对lambda做出调整时退出
        while step < self.maxIter and (lambdaPairsChanged > 0 or entireflag):
            lambdaPairsChanged = 0
            if entireflag:
                for i in range(self.m):
                    lambdaPairsChanged += self.innerLoop(i)  # 进入内循环
                step += 1
            else:
                # 提取非支持向量的数据点
                nonBoundIs = ((self.lambdas.A > 0)*(self.lambdas.A < self.C)).nonzero()[0]
                for i in nonBoundIs:
                    lambdaPairsChanged += self.innerLoop(i)  # 进入内循环
                step += 1
            if entireflag:
                entireflag = False  # 转换标志位：切换到另一种
            # 转换标志位：遍历整个数据集
            elif lambdaPairsChanged ==0:
                entireflag = True
        self.svIndex = (self.lambdas.A > 0).nonzero()[0]     # 计算完成后的支持向量索引
        self.sptVects = self.X[self.svIndex]                 # 计算完成后的支持向量
        self.SVlabel = self.labelMat[self.svIndex]           # 计算完成后的支持向量类别标签
        m, n = self.X.shape
        self.w = np.zeros((n, 1))
        for i in range(m):
            self.w += np.multiply(self.lambdas[i] * self.labelMat[i], self.X[i, :].T)
        self.labelMat = [i[0] for i in self.labelMat.tolist()]
        # self.b = self.b.tolist()[0]

    # SVMbySOM主函数内循环：innerLoop
    def innerLoop(self, i):
        self.b = 0           # 超平面截距初始值
        Ei = self.calcEk(i)  # 计算和更新i的误差缓存
        if ((self.labelMat[i]*Ei < -self.tol) and (self.lambdas[i] < self.C)) or ((self.labelMat[i]*Ei > self.tol) and (self.lambdas[i] > 0)):
            j, Ej = self.chooseJ(i, Ei)             # 选择具有最大误差的j
            # 初始化lambda1old和lambda2old
            lambdaIold = self.lambdas[i].copy()
            lambdaJold = self.lambdas[j].copy()
            if self.labelMat[i] != self.labelMat[j]:
                L = max(0, self.lambdas[j] - self.lambdas[i])
                H = min(self.C, self.C + self.lambdas[j] - self.lambdas[i])
            else:
                L = max(0, self.lambdas[j] + self.lambdas[j] - self.C)
                H = min(self.C, self.lambdas[j] + self.lambdas[i])
            if L == H:
                return 0
            eta = 2.0*self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0
            self.lambdas[j] -= self.labelMat[j]*(Ei -Ej)/eta
            self.lambdas[j] = self.clipLambda(self.lambdas[j], H, L)
            self.eCache[j] = [1, self.calcEk(j)]    # 计算和更新j的缓存
            if abs(self.lambdas[j] - lambdaJold) < 0.00001:
                return 0
            self.lambdas[i] += self.labelMat[j]*self.labelMat[i]*(lambdaJold - self.lambdas[j])
            self.eCache[i] = [1, self.calcEk(i)]    # 计算和更新j的缓存
            b1 = self.b - Ei - self.labelMat[i] * (self.lambdas[i] - lambdaIold) * self.K[i, i] - self.labelMat[j] \
                 * (self.lambdas[j] - lambdaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labelMat[i] * (self.lambdas[i] - lambdaIold) * self.K[i, j] - self.labelMat[j] \
                 * (self.lambdas[j] - lambdaJold) * self.K[j, j]
            if 0 < self.lambdas[i] and self.C > self.lambdas[i]:
                self.b = b1
            elif 0 < self.lambdas[j] and self.C > self.lambdas[j]:
                self.b = b2
            else:
                self.b = (b1+b2)/2.0
            return 1
        else:
            return 0

    # 训练数据集的预测
    def predict(self):
        self.plabelMat = []  # 储存预测类别标签
        m, n = np.mat(self.X).shape
        for i in range(m):
            kernelEval = self.kernels(np.mat(self.sptVects), self.X[i, :])
            predict = kernelEval.T * np.multiply(self.SVlabel, self.lambdas[self.svIndex]) + self.b
            self.plabelMat.append(int(np.sign(predict).tolist()[0][0]))
        self.TP = list(np.array(self.labelMat)+np.array(self.plabelMat)).count(2)
        self.FN = list(np.array(self.labelMat)-np.array(self.plabelMat)).count(2)
        self.FP = list(np.array(self.labelMat)-np.array(self.plabelMat)).count(-2)
        self.TN = list(np.array(self.labelMat)+np.array(self.plabelMat)).count(-2)
        if not (self.TP ==0 or self.FP == 0):
            self.A = (self.TP+self.TN)/len(self.labelMat)
            self.P = self.TP/(self.TP+self.FP)
            self.R = self.TP/(self.TP+self.FN)
            self.F1 = 2*self.TP/(2*self.TP+self.FP+self.FN)

    # 二类分类器
    def classify(self, testSet, testLabel):  # 测试集testSet及其分类标签testLabel
        testSet = [list(i) for i in (np.array(testSet)[:, 0:self.n]-np.array(testSet)[:, 0:self.n].mean())/np.array(testSet)[:, 0:self.n].std()]
        testSet = np.mat(testSet)
        plabelMat = []
        m, n = testSet.shape
        for i in range(m):
            kernelEval = self.kernels(np.mat(self.sptVects), testSet[i, :])
            predict = kernelEval.T * np.multiply(self.SVlabel, self.lambdas[self.svIndex]) + self.b
            plabelMat.append(int(np.sign(predict).tolist()[0][0]))
        TP = list(np.array(testLabel) + np.array(plabelMat)).count(2)
        FN = list(np.array(testLabel) - np.array(plabelMat)).count(2)
        FP = list(np.array(testLabel) - np.array(plabelMat)).count(-2)
        TN = list(np.array(testLabel) + np.array(plabelMat)).count(-2)
        self.testA = (TP+TN)/len(testLabel)
        self.testP = TP / (TP + FP)
        self.testR = TP / (TP + FN)
        self.testF1 = 2 * TP / (2 * TP + FP + FN)

    # 随机选取50个数据作为测试集。
    def chooserandomTestDataSet(self, n):
        self.TestDataSet = []
        for i in range(n):
            r = random.randrange(0, len(self.DataSet), 1)
            self.TestDataSet.append(self.DataSet.pop(r))
        self.testlabelMat = np.array(self.TestDataSet)[:, self.n]
        self.testX = [list(i) for i in (np.array(self.TestDataSet)[:, 0:self.n] - np.array(self.TestDataSet)[:, 0:self.n].mean()) / np.array(self.TestDataSet)[:, 0:self.n].std()]

    # 获取最佳惩罚因子C的值
    def getbestC(self, min, max, d):
        clist = np.linspace(min, max, d)
        plt.figure(dpi=300)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        al = []
        pl = []
        rl = []
        f1l = []
        cl = []
        for i in clist:
            self.C = i
            self.processDataSet()
            self.train()    # 训练数据
            self.predict()  # 预测类别
            if self.TP == 0 or self.FP == 0:
                continue
            else:
                A = (self.TP+self.TN)/len(self.labelMat)
                P = self.TP/(self.TP+self.FP)
                R = self.TP/(self.TP+self.FN)
                F1 = 2*self.TP/(2*self.TP+self.FP+self.FN)
                al.append(A)
                pl.append(P)
                rl.append(R)
                f1l.append(F1)
                cl.append(i)
                print('C =', i, '--(准确率)：A =', A, '(精确率):P =', P, '(召回率):R =', R, '(F1值):F1 =', F1)
        plt.plot(cl, al, marker="*")
        plt.plot(cl, pl, marker="*")
        plt.plot(cl, rl, marker="*")
        plt.plot(cl, f1l, marker="*")
        plt.xlabel('惩罚因子C')
        plt.legend(['准确率A', '精确率P', '召回率R', 'F1值'], loc='lower left')
        plt.title('getbestC!')
        plt.show()
        ca = al.copy()
        cp = pl.copy()
        cr = rl.copy()
        cf1 = f1l.copy()
        for i in range(len(cp) - 1):
            for j in range(0, len(cp) - i-1):
                if ca[j] > ca[j + 1]:
                    cp[j], cp[j + 1] = cp[j + 1], cp[j]
                    cl[j], cl[j + 1] = cl[j + 1], cl[j]
                    cr[j], cr[j + 1] = cr[j + 1], cr[j]
                    cf1[j], cf1[j + 1] = cf1[j + 1], cf1[j]
                    ca[j], ca[j + 1] = ca[j + 1], ca[j]
        print('The best C ', 'is:', cl[-1], ',with the A =', ca[-1], 'P =', cp[-1], ',R =', cr[-1], ',F1 =', cf1[-1], '\n')
        self.bestC = cl[-1]
        return self.bestC

    # S-折交叉验证
    def S_FCY(self, n):
        testal = []
        originDataSet = self.DataSet.copy()
        for i in range(n):
            self.DataSet = originDataSet.copy()
            self.chooserandomTestDataSet(60)
            self.classify(self.testX, self.testlabelMat)
            # 储存各次的训练集和测试集的准确率
            testal.append(self.testA)
        plt.figure(dpi=300)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(list(range(0, len(testal))), testal, marker="*")
        plt.xlabel('n/第n次交叉验证')
        plt.legend(['准确率'], loc='lower left')
        plt.title('S-折交叉验证')
        plt.show()
        self.DataSet = originDataSet.copy()
        print('泛化能力（准确率）：', np.array(testal).mean())

# 多分类SVM类
class MultiSVMbySMO(SVMbySMO):
    def randomsamplec(self, c):
        d1 = []
        i = 0
        while i < len(self.DataSet):
            if self.DataSet[i][-1] == c:
                self.DataSet[i][-1] = 1   # 关于第i个类的数据集。对i类的类别标签修改为1/非i类的类别标签修改为-1。
                d1.append(self.DataSet[i])
                self.DataSet.pop(i)
            else:
                self.DataSet[i][-1] = -1
                i += 1
        if len(d1) < len(self.DataSet):
            d2 = random.sample(self.DataSet, len(d1))
            self.DataSet = d1 + d2
        else:
            self.DataSet = d1 + self.DataSet

    # 数据集(异常值（缺损值）等)处理及标准化
    def processDataSet(self):
        self.labelMat = np.array(self.DataSet)[:, self.n]
        # 数据标准化
        self.X = [list(i) for i in (np.array(self.DataSet)[:, 0:self.n] - np.array(self.DataSet)[:, 0:self.n].mean()) / np.array(self.DataSet)[:, 0:self.n].std()]
        self.initparam()

    def processDataSetc(self, c):
        self.originallabelMat = list(np.array(self.DataSet)[:, self.n])
        i = 0
        while i < len(self.DataSet):
            if self.DataSet[i][-1] == c:
                self.DataSet[i][-1] = 1   # 关于第i个类的数据集。对i类的类别标签修改为1/非i类的类别标签修改为-1。(i)
            else:
                self.DataSet[i][-1] = -1
                i += 1

    # 调整惩罚因子C,获取最佳惩罚因子C的值
    def getbestC(self, DataSetPath, min, max, d):
        clist = np.linspace(min, max, d)
        ckl = []
        for ck in range(0, 5):
            plt.figure(dpi=200)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            pl = []
            rl = []
            f1l = []
            cl = []
            al = []
            svm0 = MultiSVMbySMO()
            svm0.loadDataSet(DataSetPath)
            svm0.randomsamplec(ck)
            for i in clist:
                svm0.C = i                          # 惩罚因子C
                svm0.tol = 0.001                    # 容错率
                svm0.maxIter = 500
                svm0.kValue['Gaussian'] = 3.0       # 核函数(高斯核函数值对训练的影响？参数σ越小，分的类别会越细，也就是说越容易导致过拟合；参数σ越大，分的类别会越粗，导致无法将数据区分开来。）
                svm0.processDataSet(13)
                svm0.train()                        # 训练数据
                svm0.predict()                      # 预测类别
                if self.TP == 0 or self.FP == 0:
                    continue
                else:
                    A = (self.TP+self.TN)/len(self.labelMat)
                    P = self.TP/(self.TP+self.FP)
                    R = self.TP/(self.TP+self.FN)
                    F1 = 2*self.TP/(2*self.TP+self.FP+self.FN)
                    al.append(A)
                    pl.append(P)
                    rl.append(R)
                    f1l.append(F1)
                    cl.append(i)
                    print('C =', i, '--(准确率)：A =', A, '(精确率):P =', P, '(召回率):R =', R, '(F1值):F1 =', F1)
            plt.plot(cl, pl, marker="*")
            plt.plot(cl, rl, marker="*")
            plt.plot(cl, f1l, marker="*")
            plt.xlabel('惩罚因子C')
            plt.legend(['准确率A','精确率P', '召回率R', 'F1值'], loc='lower left')
            t = '第'+str(ck)+'类'
            plt.title(t)
            cl = clist.copy()
            cp = pl.copy()
            cr = rl.copy()
            cf1 = f1l.copy()
            for i in range(len(cp)-1):
                for j in range(0, len(cp)-i-1):
                    if cp[j] > cp[j+1]:
                        cp[j], cp[j+1] = cp[j+1], cp[j]
                        cl[j], cl[j + 1] = cl[j + 1], cl[j]
                        cr[j], cr[j + 1] = cr[j + 1], cr[j]
                        cf1[j], cf1[j + 1] = cf1[j + 1], cf1[j]
            ckl.append(cl[-1])
            print('The best C of class', ck, 'is:', cl[-1], 'with P =:', cp[-1], ',R =', cr[-1], 'F1 =', cf1[-1], '\n')
            plt.show()
        return np.round(sum(ckl)/len(ckl))

    # 最小二乘回归组合各二分类器得到具有多分类能力的多分类器，并进行训练集的预测。
    def multclasspredict(self, DataSetPath, labels: list, C):   # labels-多分类类别标签
        self.Xl = []           # 储存五个二分类SVM的关键参数
        self.sptVectsl = []
        self.SVlabell = []
        self.lambdasl = []
        self.svIndexl = []
        self.bl = []
        for i in labels:
            self.DataSet = self.originDataSet
            self.processDataSetc(13, i)
            self.train()                                  # 训练数据
            self.predict()                                # 预测类别
            print('-----------------The class of', i, '---------------------')
            print('(准确率)：A =', self.A,'(精确率):P =', self.P, '(召回率):R =', self.R, '(F1值):F1 =', self.F1)
            print('The index of the Support Vectors:', self.svIndex)            # 输出支持向量的索引
            print('The Support Vectors', self.sptVects)                         # 输出支持向量（标准化后的）
            print('The number of Support Vectors:', self.sptVects.shape[0])     # 输出支持向量的个数
            print("b:", self.b)                                                 # 输出b值
            self.Xl.append(self.X)
            self.sptVectsl.append(self.sptVects)
            self.SVlabell.append(self.SVlabel)
            self.lambdasl.append(self.lambdas)
            self.svIndexl.append(self.svIndex)
            self.bl.append(self.b)
            self.labelvector.append(self.plabelMat)
        # 对分类向量最小二乘回归加权并进行最终预测
        self.labelw = np.linalg.inv(np.mat(self.labelvector).T*np.mat(self.labelvector))*np.mat(self.labelvector)*np.mat(self.OriginalLabelMat)
        self.labelb = np.array(self.OriginalLabelMat)-np.mat(np.array(self.labelvector).mean())*self.labelw
        self.MultplabelMat = (self.labelw*np.mat(self.labelvector).T+self.labelb).tolist()
        for i in range(len(self.MultplabelMat)):
            if self.MultplabelMat[i] < 0:
                self.MultplabelMat[i] = 0
            elif self.MultplabelMat[i] > 4:
                self.MultplabelMat[i] = 4
            else:
                self.MultplabelMat[i] = self.MultplabelMat[i].around()
        self.A = list(np.array(self.originallabelMat) - np.array(self.MultplabelMat)).count(0)/len(self.MultplabelMatself)
        print('多分类预测的准确率为：', self.A)

    # 第i类二分类SVM的预测值
    def predicti(self, i):
        plabelMati = []  # 储存预测类别标签
        m, n = np.mat(self.Xl[i]).shape
        for i in range(m):
            kernelEval = self.kernels(np.mat(self.sptVectsl[i]), self.Xl[i][i, :])
            predict = kernelEval.T * np.multiply(self.SVlabell[i], self.lambdasl[i][self.svIndex]) + self.bl[i]
            plabelMati.append(int(np.sign(predict).tolist()[0][0]))
        return plabelMati

    # 多类分类器
    def multclassify(self, testSet, testLabel):  # 测试集testSet及其分类标签testLabel
        testSet = np.mat(testSet)
        m, n = testSet.shape
        self.testlabelVector = []
        for i in range(len(testLabel)):
            self.testlabelVector.append(self.predicti(i))
        self.ptestlabelMat = (self.labelw*np.mat(self.testlabelVector).T+self.labelb).tolist()
        self.testA = list(np.array(self.testlabelMat) - np.array(self.ptestlabelMat)).count(0)/len(self.testlabelMat)
        print('多分类测试集的准确率为：', self.testA)

def main():
    # 二分类的训练和预测
    svm = SVMbySMO()
    svm.loadDataSet('processed.cleveland.data')
    for i in range(len(svm.DataSet)):
        if svm.DataSet[i][-1] == 0:
            svm.DataSet[i][-1] = 1      # 关于第i个类的数据集。对i类的类别标签修改为1/非i类的类别标签修改为-1。(i)
        else:
            svm.DataSet[i][-1] = -1
    svm.n = 13                          # 特征数
    svm.tol = 0.001                     # 容错率
    svm.maxIter = 500
    svm.kValue['Gaussian'] = 0.27       # 核函数(高斯核函数值对训练的影响-参数σ越小，分的类别会越细，也就是说越容易导致过拟合；参数σ越
                                        # 大，分的类别会越粗，导致无法将数据区分开来。）
    svm.chooserandomTestDataSet(50)     # 随机选取测试集
    testX = svm.testX.copy()
    testlabelMat = svm.testlabelMat.copy()
    # 训练集训练
    svm.processDataSet()
    svm.getbestC(100, 200, 50)  # 选取最佳的惩罚因子C，步长越小，效果越好。
    svm.processDataSet()        # 注：svm.getbestC()这部分由于挑选范围较大，迭代需要一定的时间。
    svm.C = svm.bestC
    # svm.C = 181.6326530612245            # 若不想运行上三步等待，则直接运行这行，这是上一行代码选取的结果即最优惩罚因子C值。
    # svm.processDataSet()
    svm.train()    # 训练数据
    svm.predict()  # 预测类别
    print('训练集：(准确率)：A =', svm.A, '(精确率):P =', svm.P, '(召回率):R =', svm.R, '(F1值):F1 =', svm.F1)
    # 测试集预测
    svm.classify(testX, testlabelMat)
    print('测试集：(准确率)：A =', svm.testA, '(精确率):P =', svm.testP, '(召回率):R =', svm.testR, '(F1值):F1 =', svm.testF1)
    # 泛化能力检验
    svm.DataSet = svm.originalDataSet
    svm.S_FCY(10)                       # S-折交叉验证（10-折）

    '''
    # 多分类的训练和预测
    multisvm = MultiSVMbySMO()
    # multisvm.getbestC('processed.cleveland.data', 100, 200, 50)
    multisvm.labelvector = []
    multisvm.C = multisvm.getbestC('processed.cleveland.data', 100, 200, 50)  # 惩罚因子C
    multisvm.tol = 0.001  # 容错率
    multisvm.maxIter = 500
    multisvm.kValue['Gaussian'] = 3.0  # 核函数(高斯核函数值对训练的影响？参数σ越小，分的类别会越细，也就是说越容易导致过拟合；参数σ越大，分的类别会越粗，导致无法将数据区分开来。）
    multisvm.loadDataSet('processd.cleveland.data')
    multisvm.chooseTestDataSet()
    multisvm.processDataSet(13)
    multisvm.multclasspredict()
    multisvm.multclassify()
    '''



if __name__ == '__main__':
    main()