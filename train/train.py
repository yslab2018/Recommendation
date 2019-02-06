import argparse
import pickle

import numpy as np
import cupy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.variable import Variable
from chainer.utils import WalkerAlias
from chainer import optimizers, serializers, cuda, initializers, training, reporter

from Custom_Evaluator import Evaluator
from Custom_Classifier import Classifier
from Converter import *
from Model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')    
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of GRU hidden units size')
    parser.add_argument('--batchsize', '-b', default=100, type=int,
                        help='learning minibatch size')
    parser.add_argument('--att_unit', '-au', default=50, type=int,
                        help='number of attention layer hidden units size')
    parser.add_argument('--att_head', '-ah', default=1, type=int,
                        help='number of attention')
    parser.add_argument('--drop_rate_input', '-dr_i', default=0.0, type=float,
                        help='dropout rate of GRU input')
    parser.add_argument('--drop_rate_hidden', '-dr_h', default=0.0, type=float,
                        help='dropout rate of GRU next input')
    parser.add_argument('--neg_sample', '-ns', default=1000, type=int,
                        help='number of negative sample')
    parser.add_argument('--learn_rate', '-lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--lossfunc', '-lf', default='XE', type=str,
                        choices=['BPR',  'BPR_COS',
                                 'TOP1', 'TOP1_COS',
                                 'XE',   'XE_COS'],
                        help='loss function')
    parser.add_argument('--top_k', '-topk', default=20, type=int,
                    help='evaluate recommendation top k')
    parser.add_argument('--output', '-out', default='Result', type=str,
                    help='output dirctory')
    parser.add_argument('--test', action='store_true', help='Use tiny datasets for quick tests')
    args = parser.parse_args()

    #=========================
    #      Data setting
    #=========================
    
    # training sequence
    with open('../input/train_data.pkl' ,'rb') as fr:
        train_data = pickle.load(fr)
        
    # development sequence
    with open('../input/eval_data.pkl' ,'rb') as fr:
    #with open('../input/dev_data.pkl' ,'rb') as fr:
        eval_data = pickle.load(fr)

    # embed_data
    embed_vec = np.load('../data/embed_data.npy')
    
    # Transition to test mode
    if args.test:
        print('\n +==================+')
        print(' | Test mode done!! |')
        print(' +==================+\n')

        train_data = train_data[:100]
        eval_data  = eval_data[:100]
        

    #=========================
    # Define model paramerter
    #=========================
    gpu_flag   = args.gpu              # GPU id 
    n_epoch    = args.epoch            # エポック数
    n_unit     = args.unit             # GRU中間層次元数
    batchsize  = args.batchsize        # バッチサイズ
    n_att_unit = args.att_unit         # Attention中間層次元数
    n_att_head = args.att_head         # Attention数
    neg_sample = args.neg_sample       # Negative sample数
    learn_rate = args.learn_rate       # Learning rate
    lossfunc   = args.lossfunc         # 損失関数の選択
    dr_input   = args.drop_rate_input  # GRU入力におけるdropout率
    dr_hidden  = args.drop_rate_hidden # GRUの前回出力におけるdropout率
    top_k      = args.top_k            # 推薦リスト内の考慮する位置 
    output_dir = args.output           # log出力ディレクトリ
    n_embed    = embed_vec.shape[1]    # 埋め込み層次元数
        
    # num of item
    n_class = 10

    # item id list
    count = [1 for i in range(n_class)]
            
    #=========================
    #      Model setting
    #=========================
    
    model = Classifier(GRU_Encoder(n_embed, n_unit, n_att_unit, n_att_head, dr_hidden, dr_input,
                                   embed_vec, gpu_flag),
                       lossfunc, neg_sample, count, top_k)
    
    
    if gpu_flag >= 0:
        chainer.cuda.get_device_from_id(gpu_flag).use()
        model.to_gpu()
    
    #=========================
    #   optimizer setting
    #=========================
    optimizer = chainer.optimizers.AdaGrad(lr=learn_rate)
    optimizer.setup(model)
    
    #=========================
    #   set up a iterater
    #=========================
    train_iter = chainer.iterators.SerialIterator(train_data, batchsize, shuffle=True)
    eval_iter  = chainer.iterators.SerialIterator(eval_data  , batchsize*5,
                                                  repeat=False, shuffle=False)
    
    #=========================
    #    Set up a trainer
    #=========================
    updater = training.StandardUpdater(train_iter, optimizer,
                                       converter=convert, device=gpu_flag)
    
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=output_dir)

    #==============================
    # set up a val data evaluator
    #==============================
    val_eval = Evaluator(eval_iter, model, convert, gpu_flag, len(eval_data), top_k)
    val_eval.default_name = 'eval'
    trainer.extend(val_eval)
    
    #==============================
    # Set up a trainer extensions
    #==============================
    trainer.extend(extensions.LogReport())   # logレポート表示
    trainer.extend(extensions.ProgressBar()) # 進捗バー表示

    # modelのスナップショットを保存
    trainer.extend(extensions.snapshot_object(model.predictor,
                                              filename='model_epoch-{.updater.epoch}'))

    # レポート内容の設定
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'eval/main/loss',
         'eval/main/Recall@{}'.format(top_k), 'eval/main/MRR@{}'.format(top_k),
         'elapsed_time']))
    
    
    #=========================
    #   done model learning
    #=========================
    trainer.run()
    
    
    
if __name__ == '__main__':
    main()
