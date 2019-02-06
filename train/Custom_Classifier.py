import numpy as np
import cupy

import chainer
import chainer.links as L
from chainer import reporter
from chainer.variable import Variable

from LossFunction import Loss
from chainer import cuda

class Classifier(L.Classifier):

    def __init__(self, predictor, lossfunc, ns, count, top_k, label_key=-1):

        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' % type(label_key))

        super(Classifier, self).__init__(predictor)

        self.lossfunc  = Loss(ns, count, lossfunc)
        self.top_k     = top_k  
        self.loss      = None
        self.label_key = label_key
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2

        x = args[0]
        t = args[1]

        self.loss = None
        out       = self.predictor(x)
        
        self.loss = self.lossfunc(out, t, self.predictor)
        reporter.report({'loss': self.loss.data}, self)
        
        if not chainer.config.train:            
            
            # calc recall@top_k and MRR@top_k
            sum_recall = 0  # Recall  
            sum_mrr    = 0  # MRR
            
            lookup_table = np.copy(self.predictor.embed)

            # array type check
            if isinstance(out.data, (np.ndarray, np.generic)):
                xp = np
            else:
                xp = cupy
                lookup_table = cuda.to_gpu(lookup_table)
            
            ranking = xp.matmul(out.data, lookup_table.T)
            
            # ==== cos similar =====
            if self.lossfunc.calc.cos_flag:
                norm1 = xp.sum(out.data * out.data, axis=1) ** 0.5
                norm1 = xp.reshape(norm1, (norm1.shape[0], 1))

                norm2 = xp.sum(lookup_table * lookup_table, axis=1) ** 0.5
                norm2 = xp.reshape(norm2,(norm2.shape[0], 1))

                ranking = ranking/norm1/norm2.T
            # ======================

            d = xp.argsort(ranking, axis=1) # sortかつindexで表示
            d = d[:,::-1]                   # 降順に変更
            
            for i, row in enumerate(d):
                
                tmp = xp.where(row[:self.top_k] == t[i])[0]
                if tmp.shape[0] != 0:
                    
                    # count Recall
                    sum_recall += 1
                    
                    # count MRR
                    rank = tmp[0] + 1
                    sum_mrr += (1 / rank)
                    
            
            reporter.report({'Sum_Recall' : sum_recall}, self)
            reporter.report({'Sum_MRR' : sum_mrr}, self)

            del lookup_table

        
        return self.loss
    
        
