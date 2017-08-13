import warnings
from asl_data import SinglesData
from asl_utils import *

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    # TODO implement the recognizer
    # return probabilities, guesses

    # Create blank lists for probabilities and X's
    probabilities = []
    guesses = []

    # So what we're going to do is get the list of x lengths.
    # We need to use the asl_utils file for this.
    x_lengths = test_set.get_all_Xlengths()

    # Then we iterate over the x values and their corresponding lengths:
    for x, length in x_lengths.values():

        # save a blank dictionary of the liklihood score:
        ld = {}

        # store minimum possible score to save max score later:
        current_best_score = float('-inf')

        # store the highest probability word, keep n/a for start.
        current_best_word = 'n/a'

        # Then we iterate over the word and the model that we developed for each word.
        # The information is stored in models within items.
        for word, model in models.items():

            # Try was added because some words don't have a return.
            try:

            # We label the score
            score = model.score(x, length)

            # we label the blank dictionary
            ld[word] = score

                # if the word's score is greater than our current max score
                if score > current_best_score:

                    # The word's score is now the max score
                    current_best_score = score
                    current_best_word = word

            # Exception for when we have no response.
            except:

                # When we have no response the liklihood is infinitely low.
                ld['word'] = float('-inf')


