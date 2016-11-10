from sklearn.base import TransformerMixin, BaseEstimator, clone
import pandas as pd

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection

    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.

    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: []
    """
    
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        trans : pandas DataFrame
            contains selected columns of X
        """
        trans = X[self.columns].copy()
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function

        Parameters
        ----------
        X : pandas DataFrame
        y : default None

        Returns
        ----------
        self
        """
        return self


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application

    Parameters
    ----------
    impute : Boolean, default False

    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]
    """

    def __init__(self, func, impute = False):
        self.func = func
        self.impute = impute
        self.series = pd.Series()

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X
        """

        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self
        """

        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).copy()
        return self


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers

    Fit several DataFrame transformers and provides a concatenated
    Data Frame

    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers

    """
    
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers

    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        concatted :  pandas DataFrame
        """

        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self : object
        """
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self


class ToDummiesTransformer(BaseEstimator, TransformerMixin):
    """ A Dataframe transformer that provide dummy variable encoding
    """

    def transform(self, X, **transformparams):
        """ Returns a dummy variable encoded version of a DataFrame

        Parameters
        ----------
        X : pandas DataFrame

        Returns
        ----------
        trans : pandas DataFrame

        """

        trans = pd.get_dummies(X).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Do nothing operation

        Returns
        ----------
        self : object
        """
        return self


class DropAllZeroTrainColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides dropping all-zero columns
    """

    def transform(self, X, **transformparams):
        """ Drops certain all-zero columns of X

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : DataFrame
        """

        trans = X.drop(self.cols_, axis=1).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Determines the all-zero columns of X

        Parameters
        ----------
        X : DataFrame
        y : not used

        Returns
        ----------
        self : object
        """

        self.cols_ = X.columns[(X==0).all()]
        return self
