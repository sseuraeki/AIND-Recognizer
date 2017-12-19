import warnings
from asl_data import SinglesData


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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # getters
    words = test_set.get_all_sequences()
    hwords = test_set.get_all_Xlengths()

    # iterate through words
    for word in words:
      # get X, length
      X, length = hwords[word]
      temp = {}
      # iterate through models
      for model_key in models:
        try:
          temp[model_key] = models[model_key].score(X, length)
        except:
          temp[model_key] = -float("inf")

      # add the scores to probabilities
      probabilities.append(temp)

    # find the model with the best score
    for instance in probabilities:
      best_score = -float("inf")
      best_model = None
      for model_key in instance:
        score = instance[model_key]
        if score > best_score:
          best_score = score
          best_model = model_key
      # add to guesses
      guesses.append(best_model)

    return probabilities, guesses
