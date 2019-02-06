import numpy as np
import cupy

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.utils import WalkerAlias
from chainer import cuda

def penalization_term(aw_matrix):
    """
    Attention weight penalization term
 
    Lin, Zhouhan, et al. 
    A structured self-attentive sentence embedding.
    arXiv preprint arXiv:1703.03130 (2017).
    """
    
    if isinstance(aw_matrix, (np.ndarray, np.generic)):
        xp = np
    else:
        xp = cupy

    matrix    = xp.einsum('ijk,ikl->ijl', xp.transpose(aw_matrix, (0,2,1)), aw_matrix)
    shape     = matrix.shape
    identity  = xp.broadcast_to(xp.identity(shape[2]),shape)
    
    norm_term = xp.sum(xp.reshape((matrix-identity)**2, (shape[0],-1)), 1)**0.5

    return norm_term
                
def calc_cos(logit, x, y):
    batchsize  = x.shape[0]
    neg_sample = y.shape[0] - batchsize
    
    norm_a = F.sum(x*x, axis=1) ** 0.5
    norm_a = F.reshape(norm_a, (batchsize, 1))
    norm_a = F.broadcast_to(norm_a, (batchsize, batchsize+neg_sample))
    
    norm_b = F.sum(y*y, axis=1) ** 0.5
    norm_b = F.reshape(norm_b, (batchsize+neg_sample, 1))
    norm_b = F.broadcast_to(norm_b, (batchsize+neg_sample, batchsize))
    
    return logit/norm_a/F.transpose(norm_b)
    

class Loss():
    
    def __init__(self, ns, count, func_name):
        self.sampler = WalkerAlias(count)
        self.neg_sample = ns
        
        if  func_name == 'XE':
            self.calc = XE(ns, cos_flag = False)
            
        elif func_name == 'XE_COS':
            self.calc = XE(ns, cos_flag = True)

        elif func_name == 'TOP1':
            self.calc = TOP1(ns, cos_flag = False)
            
        elif func_name == 'TOP1_COS':
            self.calc = TOP1(ns, cos_flag = True)

        elif func_name == 'BPR':
            self.calc = BPR(ns, cos_flag = False)
            
        elif func_name == 'BPR_COS':
            self.calc = BPR(ns, cos_flag = True)

        
    def __call__(self, x, t, predictor):

        # Negative item sampling
        sample = self.sampler.sample(self.neg_sample)
        con_t  = np.concatenate((t, sample.astype(np.int32)))
        
        y = predictor.embed[con_t]
        if not isinstance(x.data,(np.ndarray, np.generic)):
            y = cuda.to_gpu(y) 
        
        # calculate penalization term
        norm_term = penalization_term(predictor.tmp_weight)
        
        # calculate loss
        loss = self.calc(x, y)
        
        return F.average(loss+norm_term)

    



class XE():
    """
    XE (Cross entoropy)

    Hidasi, Balázs, and Alexandros Karatzoglou. 
    Recurrent Neural Networks with Top-k Gains for Session-based Recommendations.
    arXiv preprint arXiv:1706.03847 (2017).
    """
    
    def __init__(self, ns, cos_flag=False):
        self.neg_sample = ns
        self.cos_flag   = cos_flag
        
    def __call__(self, x, y):
        batchsize = x.shape[0] # B
        
        logit = F.matmul(x, y, transb=True)  # (B, B+ns)

        # cos類似度を計算 
        if self.cos_flag:
            logit = calc_cos(logit, x, y)

        # axis=1でsohtmax適用
        logit = F.softmax(logit,1)
        
        mask_E = np.identity(batchsize, np.float32)                 # (B, B)の単位行列を作成
        mask_Z = np.zeros((batchsize, self.neg_sample), np.float32) # (B, ns)の零行列を作成
        mask   = np.concatenate((mask_E, mask_Z), axis=1)           # (B, B+ns)の形へ連結 
            
        if not isinstance(x.data,(np.ndarray, np.generic)):
            mask = cuda.to_gpu(mask)
        
        gold_sim = F.sum(logit*mask, 1)    # 対角成分を抽出
        tmp      = gold_sim + 1e-32        # 0回避 
        loss     = -F.log(tmp)
        
        return loss

class TOP1():
    """
    TOP1

    Hidasi, Balázs, et al. 
    Session-based recommendations with recurrent neural networks.
    arXiv preprint arXiv:1511.06939 (2015).
    """
    
    def __init__(self, ns, cos_flag=False):
        self.neg_sample = ns
        self.cos_flag   = cos_flag
        
    def __call__(self, x, y):
        batchsize = x.shape[0] # B
        
        logit = F.matmul(x, y, transb=True)  # (B, B+ns)

        # cos類似度を計算 
        if self.cos_flag:
            logit = calc_cos(logit, x, y)
        
        mask_E = np.identity(batchsize, np.float32)                 # (B, B)の単位行列を作成
        mask_Z = np.zeros((batchsize, self.neg_sample), np.float32) # (B, ns)の零行列を作成
        mask   = np.concatenate((mask_E, mask_Z), axis=1)           # (B, B+ns)の形へ連結 
            
        if not isinstance(x.data,(np.ndarray, np.generic)):
            mask = cuda.to_gpu(mask)
        
        gold_sim = F.sum(logit*mask, 1)    # 対角成分を抽出
        gold_sim = F.broadcast_to(gold_sim, (batchsize+self.neg_sample, batchsize))
        gold_sim = F.swapaxes(gold_sim, 0, 1)

        loss = F.average(F.sigmoid((logit - gold_sim)) + F.sigmoid(logit**2),1)
        
        return loss


class BPR():
    """
    BPR (Bayesian Personalized Ranking)

    Rendle, Steffen, et al. 
    BPR: Bayesian personalized ranking from implicit feedback.
    In Proc AUAI, 2009.
    """
    
    def __init__(self, ns, cos_flag=False):
        self.neg_sample = ns
        self.cos_flag   = cos_flag
        
    def __call__(self, x, y):
        batchsize = x.shape[0] # B
        
        logit = F.matmul(x, y, transb=True)  # (B, B+ns)

        # cos類似度を計算 
        if self.cos_flag:
            logit = calc_cos(logit, x, y)
        
        mask_E = np.identity(batchsize, np.float32)                 # (B, B)の単位行列を作成
        mask_Z = np.zeros((batchsize, self.neg_sample), np.float32) # (B, ns)の零行列を作成
        mask   = np.concatenate((mask_E, mask_Z), axis=1)           # (B, B+ns)の形へ連結 
            
        if not isinstance(x.data,(np.ndarray, np.generic)):
            mask = cuda.to_gpu(mask)
        
        gold_sim = F.sum(logit*mask, 1)    # 対角成分を抽出
        gold_sim = F.broadcast_to(gold_sim, (batchsize+self.neg_sample, batchsize))
        gold_sim = F.swapaxes(gold_sim, 0, 1)
        
        tmp_loss = F.sigmoid((gold_sim-logit)) + 1e-32 # 0除去
        loss     = F.average(-F.log(tmp_loss), 1)
        
        return loss
