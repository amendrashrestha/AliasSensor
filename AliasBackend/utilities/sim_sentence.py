__author__ = 'amendrashrestha'

from gensim import models
import os


filepath = os.environ['HOME'] + '/Downloads/GoogleNews-vectors-negative300.bin'

model = models.Word2Vec.load_word2vec_format(filepath, binary=True)


s1 = 'This is actually a pretty challenging problem that you are asking. Computing sentence similarity requires building a grammatical model of the sentence, understanding equivalent structures (e.g. "he walked to the store yesterday" and "yesterday, he walked to the store"), finding similarity not just in the pronouns and verbs but also in the proper nouns, finding statistical co-occurences / relationships in lots of real textual examples, etc.'

s2 = 'The simplest thing you could try -- though I don\'t know how well this would perform and it would certainly not give you the optimal results -- would be to first remove all "stop" words (words like "the", "an", etc. that don\'t add much meaning to the sentence) and then run word2vec on the words in both sentences, sum up the vectors in the one sentence, sum up the vectors in the other sentence, and then find the difference between the sums. By summing them up instead of doing a word-wise difference, you\'ll at least not be subject to word order. That being said, this will fail in lots of ways and isn\'t a good solution by any means (though good solutions to this problem almost always involve some amount of NLP, machine learning, and other cleverness).'

#calculate distance between two sentences using WMD algorithm
distance = model.wmdistance(s1, s2)

print ('distance = %.3f' % distance)

