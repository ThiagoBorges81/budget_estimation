import inflection
from sklearn.base import BaseEstimator, TransformerMixin



class FormatRawColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.rename_columns(X)

        return X
    
    def rename_columns(self, dataframe):
        # Copy the data frame
        df = dataframe.copy()

        # List the columns names
        cols_raw = list(df.columns)

        # Convert headers to snakecase format
        snakecase = lambda x: inflection.underscore( x )

        # Implement new column names
        cols_new = list( map( snakecase, cols_raw ) )

        # replace the dataframe column names with the newly formatted names
        df.columns = cols_new

        return df