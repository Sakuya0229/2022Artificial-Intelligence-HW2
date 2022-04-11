import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        single_list = []
        bi_list = []
        single_dic = Counter()
        for s in tqdm(corpus_tokenize):
            for token in s:
                if token not in single_list:
                    single_list.append(token)
                single_dic[token]+=1
            for i in range(0,len(s)-1,1):
                bi_list.append((s[i],s[i+1]))
        # print(bi_list)
        bi_dic = Counter()
        # for s in tqdm(corpus_tokenize):
            # strtmp = ' '.join(str(w) for w in s)
        for bi in bi_list:
            bi_dic[bi] += 1
                # print(bi)
                # num = str(s).count(bi)
                # bi_dic[bi] += num
        # print(bi_dic)
        listOfProb = {}
        for bigram in bi_list:
            word1 = bigram[0]
            word2 = bigram[1]
            listOfProb[bigram] = (bi_dic.get(bigram))/(single_dic.get(word1))
        # print(listOfProb)
        dic_ans={}
        for token in tqdm(single_list):
            dic_tmp = {}
            for key in listOfProb.keys():
                if key[0] == token:
                    dic_tmp.setdefault(key[1],listOfProb[key])
            dic_ans.setdefault(token,dic_tmp)

        # print(dic_ans)

        return dic_ans,listOfProb,bi_dic

        # end your code

    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.listOfProb,self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]

        # begin your code (Part 2)
        bi_list = []
        for s in tqdm(corpus):
            for i in range(0,len(s)-1,1):
                bi_list.append((s[i],s[i+1]))
        listOfProb = self.listOfProb
        tmp = 0
        for bi in bi_list:
            # print(listOfProb[bi])
            tmp += math.log(listOfProb.get(bi) if listOfProb.get(bi) != None else 0.00001,2)
        perplexity = 2**(-(tmp/len(bi_list)))
        # print(perplexity)
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)

        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 250
        feature_dic = self.features
        # print(feature_dic)
        tmp_list = sorted(feature_dic.items(),key = lambda item:item[1],reverse = True)
        tmp_list = tmp_list[:feature_num]
        feature_list =[]
        for l in tmp_list:
            feature_list.append(l[0])
        # print(feature_list)

        train_corpus_embedding =[]
        corpus_train = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        for s in tqdm(corpus_train):
            train_bi_list = []
            train_bi_dic = Counter()
            sen_tmp_list = []
            for i in range(0,len(s)-1,1):
                train_bi_list.append((s[i],s[i+1]))
            for bi in train_bi_list:
                train_bi_dic[bi] += 1
            for feature in feature_list:
                sen_tmp_list.append(train_bi_dic.get(feature) if train_bi_dic.get(feature)!= None else 0 )
            train_corpus_embedding.append(sen_tmp_list)
        # print(train_corpus_embedding)

        test_corpus_embedding =[]
        corpus_test = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        for s in tqdm(corpus_test):
            test_bi_list = []
            test_bi_dic = Counter()
            sen_tmp_list = []
            for i in range(0,len(s)-1,1):
                test_bi_list.append((s[i],s[i+1]))
            for bi in test_bi_list:
                test_bi_dic[bi] += 1
            for feature in feature_list:
                sen_tmp_list.append(test_bi_dic.get(feature) if test_bi_dic.get(feature)!= None else 0 )
            test_corpus_embedding.append(sen_tmp_list)
        # print(test_corpus_embedding)

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.

        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
    # model.train_sentiment(test_sentence, test_sentence)
