import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
# lol
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
        # This would be more helpful:
        self.n_components = range(self.min_n_components, self.max_n_components + 1)
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
    Akaike information criteria: AIC = -2 * logL + 2 * p
    Better model < less better model
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # TODO implement model selection based on BIC scores

        # Store a list of BIC's
        BICs = []


        # For each "N" in our component list:
        for n in self.n_components:

            # Apparently there are errors in the data
            try:

                # Grab the base model:
                base_mod = self.base_model(n)

                # Grab the model's score (LD / LogL):
                ld = base_mod.score(self.X, self.lengths)

                # Grab the model n features to calculate parameters later:
                model_features = model.n_features

                # Calculate number of parameters:
                num_params = (n ** 2) + ((2 * (model_features - 1) * n ))

                # Get BIC using equation:
                BIC = (-2 * ld) + (num_params * math.log(n))

                BICs.append(BIC)

            # error deal
            except Exception as error:
                pass

        # Generating output
        # Sometimes we have no BICs list
        if BICs:

            # Get out maximum BIC
            output = n_components[np.argmin(BICs)]

        else:
            output = self.n_constant

        return self.base_model(output)




class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # A grader helped me with this code with some suggestions.
        try:
            # Create best_dic & model fillers
            best_dic = float("-Inf")
            best_model = None

            # Begin for loop to go over items in n_components
            for n in self.n_components:

                # Grab the model for this step
                model = self.base_model(n)

                # Grab the LogL for this model
                log_L = model.score(self.X, self.lengths)

                # Getting the penalty
                """
                This needs a little bit more explanation.
                It is a list comprehension statement that looks through the mean score
                Each word is evaluated as long as it is not equal to the current word.
                """
                penalty = np.mean([model.score(self.hwords[word]) for word in self.words if word != self.this_word])

                #Calculate DIC
                dic = log_L - penalty

                #If it's better than our current best score then it becomes the best score.
                if dic > best_dic:
                    best_model = model
                    best_dic = dic

            # Return our best model
            return best_model

        except Exception as error:

            best_model = self.base_model(self.n_constant)

            return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):

        # So for this we need to select the best model
        # The best model has the highest mean value
        # But we need to create a list for that value

        # Make a blank list with no values:
        mean_values = []

        # This is given, this is not mine:
        split_method = KFold()

        # Try again
        try:

            # For each n in n components:
            for n in self.n_components:

                # We get the base model
                base_mod = self.base_model(n)

                # We need to make a loop within the loop to calculate the value
                # of each fold:

                # Blank fold value list:
                fold_values = []

                # We need to get the test sequences.
                # We'll only be interested in j here to recombine
                # the sequence after splitting the originals

                # I got this from the asl_recognizer, it is not my idea:

                """
                **Tip:** In order to run `hmmlearn` training using the X,lengths tuples on the new folds, 
                subsets must be combined based on the indices given for the folds.  A helper utility has been 
                provided in the `asl_utils` module named `combine_sequences` for this purpose.
                """
                for i, j in split_method.split(self.sequences):

                    # I mentioned this above
                    x, length = combine_sequences(i, self.sequences)

                    # This gets our model data.
                    model = GaussianHMM(n_components = num_states, covariance_type = "diag", n_iter = 1000, random_state = self.random_state, verbose = False).fit(x, length)

                    # Now we can just append to the fold values
                    fold_values.append(model.score(x, length))

                # Now we can just append the mean values using the mean of the fold values
                mean_values.append(np.mean(fold_values))

        # exception
        except Exception as error:
            pass

        # Same scenario as the DICs
        if mean_values:
            output = self.base_model(self.n_components[np.argmax(mean_values)])

        else:
            output = self.base_model(self.n_constant)

        return output







