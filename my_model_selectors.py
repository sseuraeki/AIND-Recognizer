import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score = float("inf")
        best_model = None

        # iteration through n values
        for n in range(self.min_n_components, self.max_n_components):
            try:
                # define model
                model = GaussianHMM(n_components=n,
                                    random_state=self.random_state).fit(self.X, self.lengths)

                # now to compute the BIC score
                # according to 'http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf'
                # BIC = -2 * log(L) + p * log(N)
                # where L is the likelihood, p is the number of parameters,
                # and N is the number of data points

                # likelihood score
                # according to the hmmlearn library document,
                # score() returns log-likelihood score
                logL = model.score(self.X, self.lengths)

                # number of parameters
                # this should be the sum of following:
                # number of (transition probs, starting probs, means, variances)
                # according to 'http://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GaussianHMM'
                # we can get the shape of each array
                tran_probs = model.n_components * model.n_components
                start_probs = model.n_components
                means = model.n_components * model.n_features
                # since the default model covariance type is 'diag',
                # according to the hmmlearn document,
                # the covar array shape should be (n_components, n_features, n_features)
                variances = model.n_components * model.n_features * model.n_features
                p = tran_probs + start_probs + means + variances

                # number of data points
                N = self.X.shape[0]

                # compute BIC
                BIC = -2 * logL + p * np.log(N)

                # BIC is lower the better
                # because p*np.log(N) gets larger as the model gets more complicated
                if BIC < best_score: 
                    best_score = BIC
                    best_model = model
            except:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        best_score = -float("inf")
        best_model = None

        # iteration through n values
        for n in range(self.min_n_components, self.max_n_components):
            try:
                # define the model
                model = GaussianHMM(n_components=n,
                                    random_state=self.random_state).fit(self.X, self.lengths)
                # log(P(X(i)))
                first_term = model.score(self.X, self.lengths)
                # 1/(M-1)SUM(log(P(X(all but i))))
                second_term = 0
                for word in self.words:
                    if word != self.this_word:
                        word_, length_ = self.hwords[word]
                        second_term += model.score(word_, length_)
                second_term = second_term / (len(self.words) - 1)

                score = first_term - second_term

                # DIC's second term is penalty that gets bigger
                # as the model gets more complicated
                # since it's -second_term, DIC is bigger the better
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV       
        split_method = KFold()
        best_score = -float("inf")
        best_model = None

        # if the length of sequences is less than 3,
        # it's meaningless to do KFold so here's an exception
        if len(self.sequences) < 3:
            # iterate through given components arguments
            for n in range(self.min_n_components, self.max_n_components):
                try:
                    model = GaussianHMM(n_components=n, random_state=self.random_state)
                    model = model.fit(self.X, self.lengths)
                    score = model.score(self.X, self.lengths)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    pass
            return best_model

        # else (length of sequences is 3 or longer)
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            # get training sample
            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
            # get testing sample
            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

            # iterate through given components
            for n in range(self.min_n_components, self.max_n_components):
                try:
                    model = GaussianHMM(n_components=n, random_state=self.random_state)
                    # train on training samples
                    model = model.fit(train_X, train_lengths)
                    # test on testing samples
                    score = model.score(test_X, test_lengths)
                    # score is just a simple log-likelihood score
                    # so it's bigger the better
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    pass

        return best_model


