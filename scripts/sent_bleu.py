import nltk
import sys

reffn = sys.argv[1]
basefn = sys.argv[2]
hypofn = sys.argv[3]

fref = open(reffn, 'r', encoding='utf-8')
fbase = open(basefn, 'r', encoding='utf-8')
fhypo = open(hypofn, 'r', encoding='utf-8')

for i, (ref, base, hypo) in enumerate(zip(fref, fbase, fhypo)):
    ref = ref.split()
    hypo = hypo.split()
    base = base.split()
    base_bleu = nltk.translate.bleu_score.sentence_bleu([ref], base)
    hypo_bleu = nltk.translate.bleu_score.sentence_bleu([ref], hypo)
    if hypo_bleu > base_bleu:
        print(i)

fref.close()
fbase.close()
fhypo.close()

# hypothesis = ['This', 'is', 'cat'] 
# reference = ['This', 'is', 'a', 'cat']
# BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
# print(BLEUscore)
