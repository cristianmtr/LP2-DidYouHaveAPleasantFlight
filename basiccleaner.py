import re
import string

from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


class BasicCleaner(BaseEstimator, TransformerMixin):
    def tweet_to_words(self, raw_tweet):
        letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stops]
        return(" ".join(meaningful_words))

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.tweet_to_words)
        return clean_X
