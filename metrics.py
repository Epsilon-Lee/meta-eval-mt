'''
This is an entry file for calling all different interfaces for
the following metrics we consider in this project.
- BLEU, sentBLEU, NIST, METEOR, chrF, chrF+
- WER, TER, PER, CDER
'''

import nltk.translate.bleu_score as bleu_score
import nltk.translate.nist_score as nist_score
import nltk.translate.chrf_score as chrf_score

from jiwer import wer

# ----- String matching based -----

def sent_BLEU(reference, hypothesis):
    '''
    reference:
        is a list of string lists, if there are four string lists,
        it has four references or if there is one string list, it has only
        one reference.
    hypothesis:
        is a string list of words in the hypothesis.
    '''
    return bleu_score.sentence_bleu(
                reference,
                hypothesis
            )


def corpus_BLEU(references, hypotheses):
    '''
    references:
        [[], [], ...], within each [] before, is a list of string lists
        whose length represents the number of available reference of that
        hypothesis.
    hypotheses:
        [[], [], ...], within each [], is the a list of words of every
        hypothesis.
    '''
    return bleu_score.corpus_bleu(
                references,
                hypotheses
            )


def sent_NIST(reference, hypothesis):
    '''
    Same philosophy as with sentence BLEU.
    '''
    return nist_score.sentence_nist(
                reference,
                hypothesis
            )


def corpus_NIST(references, hypotheses):
    '''
    Same philosophy as with corpus BLEU.
    '''
    return nist_score.corpus_nist(
                references,
                hypotheses
            )


def sent_METEOR():
    '''
    '''
    pass


def corpus_METEOR():
    '''
    Just arithmic average of sentence-level METEOR scores.
    '''
    pass


def sent_chrF_plus(reference, hypothesis):
    '''
    reference:
        a list of words of the reference.
    hypothesis:
        a list of words of the hypothesis.
    '''
    return chrf_score.sentence_chrf(
        reference,
        hypothesis
    )


def corpus_chrF_plus(references, hypotheses):
    '''
    Macro-average of sentence-level chrF+ scores.
    '''
    return chrf_score.corpus_chrf(
        references,
        hypotheses
    )

# ----- Edit-based -----

def sent_WER(reference, hypothesis):
    '''
    '''
    return wer(reference, hypothesis)


def corpus_WER():
    '''
    Macro-average of sentence-level chrF+ scores.
    '''
    pass


def sent_TER():
    '''
    '''
    pass


def corpus_TER():
    '''
    '''
    pass


def sent_PER():
    '''
    '''
    pass


def corpus_PER():
    '''
    '''
    pass


def sent_CDER():
    '''
    '''
    pass


def corpus_CDER():
    '''
    '''
    pass


# ----- Unit Test -----

test_ref = 'i have a nice trip to the essential city of China'
test_ref1 = 'he has a nice trip to essential cities of China'
test_hyp = 'he have a nice trip to the essential cities of China'

test_ref_b = 'let us finish the rest of the job'
test_hyp_b = 'let\'s finish the rest of job'

print(
    'Sentence-level BLEU (without smoothing):',
    sent_BLEU([test_ref.split()], test_hyp.split())
)

print(
    'Sentence-level BLEU (without smoothing, two references):',
    sent_BLEU([test_ref.split(), test_ref1.split()], test_hyp.split())
)

print(
    'Corpus-level BLEU:',
    corpus_BLEU([[test_ref.split(), test_ref1.split()], [test_ref_b.split()]], [test_hyp.split(), test_hyp_b.split()]),
    'which is different from two arithmic average of sentence-level BLEU:',
    (sent_BLEU([test_ref.split(), test_ref1.split()], test_hyp.split()) + sent_BLEU([test_ref_b.split()], test_hyp_b.split())) / 2
)

print('\n\n')

print(
    'Sentence-level NIST:',
    sent_NIST([test_ref.split(), test_ref1.split()], test_hyp.split())
)

print(
    'Corpus-level NIST:',
    corpus_NIST([[test_ref.split(), test_ref1.split()], [test_ref_b.split()]], [test_hyp.split(), test_hyp_b.split()]),
    'which is different from two arithmic average of sentence-level NIST:',
    (sent_NIST([test_ref.split(), test_ref1.split()], test_hyp.split()) + sent_NIST([test_ref_b.split()], test_hyp_b.split())) / 2
)

print('\n\n')

test_ref_c = 'we all have a lovely tender dream'
# test_hyp_c = 'we all have a lovely tender dream'
test_hyp_c = 'i do not want to leave her alone'
print(
    'Sentence-level WER:',
    sent_WER(test_ref.split(), test_hyp.split())
)

print(
    'Sentence-level WER:',
    sent_WER(test_ref_c.split(), test_hyp_c.split())
)

print('\n\n')

print(
    'Sentence-level chrF:',
    sent_chrF_plus(test_ref.split(), test_hyp.split())
)

print(
    'Corpus-level chrF:',
    corpus_chrF_plus([test_ref.split(), test_ref_b.split()], [test_hyp.split(), test_hyp_b.split()])
)
print(
    'Corpus-level chrF:',
    corpus_chrF_plus([test_ref.split(), test_ref_b.split(), test_ref_c.split()], [test_hyp.split(), test_hyp_b.split(), test_hyp_c.split()])
)
