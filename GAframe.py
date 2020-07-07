import logging
import numpy as np
from functools import partial

"""
    [Op]
    * 交叉 : ENDX
    * 変異 : NDM
    * 交代 : MGG
    が実装してあるので、これを組み合わせて所望の遺伝的アルゴを作る

"""


class GeneticAlgorithm:

    def __init__(self, searchMax=True):
        # Trueで最大値, Falseで最小値を探す.
        self.fitness = -np.inf*(2*searchMax-1) #現行の最適値
        self.bestParam = np.array([]) #現行の最適Parameter
        self.searchMax = searchMax
        self.logger = logging.getLogger()


    def initializePoints(self, n_parents, value_range, int_index=[]):
        """ 個体を一様分布に従い n_parents点生成.
        Args:
            n_parents (int): 個体数
            value_range (list): 各パラメタ探索範囲. (e.g. [[min1,max1], [min2,max2], ... ])
            int_index (list): 離散値パラメタの列番号
        副作用:
            self.n_parents, self.value_range, self.points, self.n_dim
            self.m, self.alpha, self.beta
        Return:
            None
        """
        # n_parents点の初期点を作成
        self.int_index = int_index #valueFunc内で設定しても良い
        self.n_parents = n_parents #デフォルトでは15*ndim
        self.ndim = len(value_range)
        self.value_range = np.array(value_range)
        self.points = np.random.uniform(  # [#points, #dim]のndarray
            low=self.value_range[:,0], high=self.value_range[:,1],
            size=[n_parents, self.ndim]
        )
        self.points[:, self.int_index] = self.points[:, self.int_index].round()
        # ヒューリスティックから学習のパラメタm,alpha,betaを設定
        self.m = self.ndim + 2
        self.alpha = 0.434
        self.beta  = 0.35/np.sqrt(self.m-3)
        self.gamma = 0.35/np.sqrt(self.m-2)
        # fitness,paramsを初期化
        self.fitness = -np.inf*(2*self.searchMax-1) #現行の最適値
        self.bestParam = np.array([]) #現行の最適Parameter


    def setValueFunc(self, function, **params):
        """ 生の評価関数fにparamsを部分適用して評価関数を作成.
            f(x|params) -> self.f(x)
        Args:
            function (func): 評価関数
            params (kwargs): fのうち部分適用したい引数. keyword指定して渡す.
        副作用: self.f
        Return: None
        """
        self.f = partial(function, **params)


    def evaluate(self, parameters):
        """ parametersに対応する評価値をself.fで計算し返す.
        Args   : parameters (ndarray), shape=(#children, #params)
        副作用  : None
        Return : valuation (ndarray), shape=(#children,)
        """
        valuation = np.zeros(parameters.shape[0])
        for i,param in enumerate(parameters):
            valuation[i] = self.f(*param)
        return valuation


    def ENDX(self, n_child):
        """ ENDX方式で[交叉]. 現生世代からn_child個の子集団を生成して出力.
        Args    : n_child (int)
        副作用   : None
        Return  :
            children   (ndarray): shape=(#child+2, #parameters)
            choice_idx (ndarray): shape=(#child+2, )
        """
        ## 子の作成
        choice_idx = np.random.choice(self.n_parents, size=self.m, replace=False)
        parents = self.points[choice_idx] # m点選択
        p = parents[0:2].mean(axis=0).reshape(1, -1) # 論文式 第1項 p = (p1+p2) / 2
        # 子の数がn_childあることに注意. これの計算はbroadcastでまとめて行う
        d = (parents[1] - parents[0]).reshape(1, -1)
        term2 = np.random.normal(size=[n_child, 1], scale=self.alpha) * d # 論文式 第2項 xi*d
        p_i_prime = parents[2:] - parents[2:].mean(axis=0).reshape(1, -1) #[[],[],[]] 子を子集団の重心から測った位置: [m-2, ndim]
        term3 = (p_i_prime * np.random.normal(size=[n_child, self.m-2, 1], scale=self.beta)).sum(axis=1) #size:[n_child, m-2, 1]をかけてsum(axis=1) -> [n_child, ndim]
        children = p + term2 + term3 # ndarray:[n_child, ndim]
        # value_rangeをはみ出たものは切り詰める
        children = children*(children > self.value_range[:,0].reshape(1,-1)) + (children<=self.value_range[:,0].reshape(1,-1))*self.value_range[:,0] #min一括処理
        children = children*(children < self.value_range[:,1].reshape(1,-1)) + (children>=self.value_range[:,1].reshape(1,-1))*self.value_range[:,1] #max一括処理
        # 親1,2と結合
        children = np.append( parents[:2], children, axis=0 )
        ## 整形
        # intの入力に当たるカラムを丸める
        children[:, self.int_index] = children[:, self.int_index].round()
        return (children, choice_idx)


    def NDM(self, n_child):
        """ NDM方式で[変異]. 現生世代からn_child個の子集団と子集団の親集団内でのindexを生成して出力.
        Args   : n_child (int)
        副作用  : None
        """
        ## 子の作成
        choice_idx = np.random.choice(self.n_parents, size=self.m, replace=False)
        parents = self.points[choice_idx] # m点選択
        p_i_prime = parents[1:] - parents[1:].mean(axis=0).reshape(1, -1) #[[],[],[]], 重心から測った位置
        term2 = (p_i_prime * np.random.normal(size=[n_child, self.m-1, 1], scale=self.gamma)).sum(axis=1) #size:[n_child, m-2, 1]をかけてsum(axis=1) -> [n_child, ndim]
        children = parents[0].reshape(1, -1) + term2 # ndarray:[n_child, ndim]
        # value_rangeをはみ出たものは切り詰める
        children = children*(children > self.value_range[:,0].reshape(1,-1)) + (children<=self.value_range[:,0].reshape(1,-1))*self.value_range[:,0] #min一括処理
        children = children*(children < self.value_range[:,1].reshape(1,-1)) + (children>=self.value_range[:,1].reshape(1,-1))*self.value_range[:,1] #max一括処理
        # 親1と結合
        children = np.append( parents[0].reshape(1,-1), children, axis=0 )
        ## 整形
        # intの入力に当たるカラムを丸める
        children[:, self.int_index] = children[:, self.int_index].round()
        return (children, choice_idx)


    def MGG(self, children, choice_idx, n_replace=2):
        """ 子集団と子集団の親集団中でのindexを渡し, 評価値を元にMGG方式で[世代交代]を行う.
        Args    : (ndarray, ndarray, int). n_repalceは交換される親の個数.
        副作用   : self.points, self.fitness, self.bestParam
        Return  : None
        """
        valuation = self.evaluate(children) * (2*self.searchMax-1)
        # 最良個体と親1を交代
        argmax = np.argmax(valuation)
        p1 = children[ argmax ]
        self.points[ choice_idx[0] ] = p1
        # ランダムな個体と親2を交代
        if n_replace > 1:
            children = np.delete(children, argmax, axis=0)
            p2 = children[ np.random.choice(children.shape[0]) ]
            self.points[ choice_idx[1] ] = p2
        # 現行のparameterでもっとも良かったものを更新
        if valuation.max() > self.fitness*(2*self.searchMax-1):
            self.fitness = valuation.max() * (2*self.searchMax-1) #ここに入るのは元の関数値
            self.bestParam = p1


    def DIDC(self, n_child, N_epoch=3000):
        """ NDM,ENDXオペレータを用いてDIDCを実行.
        Args:
            n_child (int): 50がデフォらしい.
            N_epoch (int): イテレーション回数の上限
        副作用  : self.fitness, self.bestParam
        Returns:
            self.bestParam:
            self.fitness:
            fitness (ndarray):
        """
        N_stop  = self.n_parents*4  #このstepたってもbestParamの更新がない場合break
        Ts      = self.n_parents*3  #Ts時刻たっても更新がない場合にENDX
        epoch_Updated = 0 #最終更新時刻
        _fitness = np.zeros(N_epoch)
        for epoch in range(N_epoch):
            if epoch%100==0: self.logger.debug("EPOCH : %s"%epoch)
            if epoch - epoch_Updated > Ts:
                # 最良値の更新が古い: ENDX(凝集)
                children, idx = self.ENDX(n_child=n_child)
                self.MGG(children, idx, n_replace=2)
            else:
                # 最良値の更新が近い: NDM(維持&拡大)
                children, idx = self.NDM(n_child=n_child)
                self.MGG(children, idx, n_replace=1)

            # 記録
            _fitness[epoch] = self.fitness
            if _fitness[epoch-1]*(2*self.searchMax-1) < _fitness[epoch]*(2*self.searchMax-1):
                epoch_Updated = epoch
                self.logger.debug("Parameter Updated at EPOCH : %s"%epoch)
            # 中断
            if epoch - epoch_Updated > N_stop:
                print("break @%s"%epoch)
                break
        _fitness = _fitness[:epoch]
        return [self.bestParam, self.fitness, _fitness]