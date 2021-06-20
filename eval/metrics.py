__author__ = 'sony-w'
__version__ = '1.0'

from .bleu.bleu import Bleu
from .cider.cider import Cider
from .rouge.rouge import Rouge
from .spice.spice import Spice
from .meteor.meteor import Meteor

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def bleu(gts, res, verbose=1):
    b = Bleu()
    return b.compute_score(gts, res, verbose)

def cider(gts, res):
    c = Cider()
    return c.compute_score(gts, res)

def rouge(gts, res):
    r = Rouge()
    return r.compute_score(gts, res)

def spice(gts, res):
    s = Spice()
    return s.compute_score(gts, res)

def meteor(gts, res):
    m = Meteor()
    return m.compute_score(gts, res)


def bleu_score_fn(method_no:int=4, ref_type='corpus'):
    
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')
    
    def bleu_score_corpus(reference_corpus:list, candidate_corpus:list, n:int = 4):
        
        weights = [1 / n] * n
        return corpus_bleu(reference_corpus, candidate_corpus,
                           smoothing_function=smoothing_method, weights=weights)
    
    def bleu_score_sentence(reference_sentences:list, candidate_sentence:list, n:int = 4):
        
        weights = [1 / n] * n
        return sentence_bleu(reference_sentences, candidate_sentence,
                             smoothing_function=smoothing_method, weights=weights)

    
    if ref_type == 'corpus':
        return bleu_score_corpus
    
    return bleu_score_sentence
