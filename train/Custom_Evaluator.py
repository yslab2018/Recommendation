from chainer import reporter as reporter_module
from chainer.training import extensions
import copy

import warnings

import six

from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import iterators
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension

class Evaluator(extensions.Evaluator):

    def __init__(self, iterator, target, converter, device, n_eval, top_k,
                 eval_hook=None, eval_func=None):

        super(Evaluator, self).__init__(
        iterator, target, converter, device, eval_hook=None, eval_func=None)

        self.n_eval = n_eval
        self.top_k  = top_k
        

    def evaluate(self):
        
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

            
        summary = reporter_module.DictSummary()

        sum_rec = 0
        sum_mrr = 0
        
        for batch in it:
            observation = {}

            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)

                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            #
            summary.add(observation)

            sum_rec += observation['eval/main/Sum_Recall']
            sum_mrr += observation['eval/main/Sum_MRR']
            

        
        summary.add({'eval/main/Recall@{}'.format(self.top_k) : sum_rec/self.n_eval})
        summary.add({'eval/main/MRR@{}'.format(self.top_k) : sum_mrr/self.n_eval})
        
        return summary.compute_mean()
