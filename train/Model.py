import numpy as np
import cupy

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.functions as F
from chainer import cuda

class GRU_Encoder(chainer.Chain):

    def __init__(self, n_embed, n_unit, n_att_unit, n_att_head, dr_hideen, dr_input,
                 pre_vec, gpu_flag):
        initializer1 = chainer.initializers.Uniform()
        initializer2 = chainer.initializers.HeNormal()        
        
        super(GRU_Encoder, self).__init__()
        with self.init_scope():
            self.gru    = L.NStepBiGRU(1, n_embed, n_unit, dr_hideen)
            self.decode = L.Linear(None, n_embed, initialW=initializer1)
            self.att_w1 = L.Linear(n_unit*2, n_att_unit, nobias=True, initialW=initializer1)
            self.att_w2 = L.Linear(n_att_unit, n_att_head, nobias=True, initialW=initializer1)

            for w in self.gru.namedparams():
                name = w[0].split('/')[2]
                if 'w' in name:
                    shape = w[1].shape
                    w[1].initializer = initializer2
                    w[1].initialize(shape)
                    
        
        self.embed      = np.copy(pre_vec)
        self.gpu_flag   = gpu_flag
        self.dr_input   = dr_input
        self.n_att_head = n_att_head
        self.tmp_weight = None
        
    def __call__(self, x):
        x_len = [len(d) for d in x]
        x_section = np.cumsum(x_len[:-1])
        
        
        ex  = np.copy(self.embed[F.concat(x, axis=0).data])
        if self.gpu_flag >= 0:
            ex = cuda.to_gpu(ex)
        
        ex  = F.dropout(ex, ratio=self.dr_input)
        exs = F.split_axis(ex, x_section, 0)
        
        _, y_seq_list = self.gru(None, exs)
        
        #==================
        #  Self-Attention
        #==================
        # バッチ内の各ステップ出力を0でパディング 
        pd_y_seq = F.pad_sequence(y_seq_list, padding=0)            # [Batch, max_len, Unit]

        n_B, n_ML, n_U = pd_y_seq.shape
        n_AH = self.n_att_head
        
        pd_y_seq = F.reshape(pd_y_seq, (n_B, n_ML, n_U, 1))         # [Batch, max_len, Unit, 1]
        pd_y_seq = F.broadcast_to(pd_y_seq, (n_B, n_ML, n_U, n_AH)) # [Batch, max_len, Unit, Head]
        
        # concatanate
        att_in = F.concat(y_seq_list, axis=0)                       # [All element, Unit] 
        att_h1 = F.tanh(self.att_w1(att_in))                        # [All element, Att unit] 
        att_h2 = self.att_w2(att_h1)                                # [All element, Head] 
        
        att_h2_seq = F.split_axis(att_h2, x_section, 0, 0)       
        att_h2_pad = F.pad_sequence(att_h2_seq, padding=-1024.0)    # [Batch, max_len, Head]
        
        # Softmax
        weight = F.softmax(att_h2_pad, axis=1)                      # [Batch, max_len, Head]
        self.tmp_weight = weight.data
        weight = F.reshape(weight, (n_B, n_ML, 1, n_AH))            # [Batch, max_len, 1, Head]
        weight = F.broadcast_to(weight, (n_B, n_ML, n_U, n_AH))     # [Batch, max_len, Unit, Head] 
        
        # 加重和を計算
        att_out = F.sum(pd_y_seq * weight, axis=1)                  # [Batch, Unit, Head]
        out     = F.tanh(self.decode(att_out))
        
        return out

