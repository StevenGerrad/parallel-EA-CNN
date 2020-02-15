from __future__ import division
import numpy as np
class Selection(object):
    def RouletteSelection(self, _a, k):
        '''
        轮盘赌选择算法
        '''
        # array和asarray都可将结构数据转换为ndarray类型。
        # 但当数据源是ndarray时，array仍会copy出一个副本，占用新的内存，但asarray不会。
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index

if __name__ == '__main__':
    s = Selection()
    a = [1, 3, 2, 1, 4, 4, 5]
    selected_index = s.RouletteSelection(a, k=20)

    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)






