from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TextlengthExtractor(BaseEstimator, TransformerMixin):
      

    def extract_text_length(self, text):
        length_text = len(text)
        return length_text

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.extract_text_length)
        return pd.DataFrame(X_tagged)

