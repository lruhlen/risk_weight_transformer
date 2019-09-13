"""Risk weight transformer"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# pylint: disable=invalid-name


class RiskWeightTransformer(BaseEstimator, TransformerMixin):
    """Transform categorical values to conditional probabilities with respect to a target value.
    This transformer converts
    categorical variables to numeric ones without creating myriad new
    columns (e.g. one hot encoding) or enforcing distance measures that
    don't reflect reality (e.g. alphabetic order ranking).
    This method replaces a categoric value with the conditional probability
    that its value is equal to a target variable.
    What does that even mean?  Here's an example:
    You have a data set with two columns: "Fur Color", and "Is a Moose?"
    (Imagine it's a list of wildlife observations that include a handful
    of moose sightings.)  Observed fur colors include grey, brown, and red.
    In this example, "Fur Color" is the categoric variable, and "Is a Moose?"
    is the target variable.
    There are 10 observations of red-furred animals.  None of those were moose.
    There are 5 observations of grey-furred animals.  One of those was a moose.
    There are 4 observations of brown-furred animals.  Three of those were moose.
    The risk-weight (of being a moose) for each of the observed fur colors is:
        + Red: 0 / 10 = 0.0
        + Grey: 1 / 5 = 0.2
        + Brown: 3 / 4 = 0.75
    In subsequent observations, we replace "fur color = grey" with
    "is-a-moose risk weight = 0.2", and so on for the other fur colors
    present in the training data set.
    How do we handle novel fur colors?  If a pink critter crosses our radar,
    we won't know (based on historic data) its true conditional likelihood
    of being a moose.
    We can decide ahead of time what risk-weight value we'll default to in
    such situations.  In this implementation, we default to the average
    of all known risk weights: (0.0 + 0.2 + 0.75) / 3.0 ~= 0.32.  Other defaults
    may make sense in other situations.  This particular default is rather
    'paranoid' that everything is a moose, which best suits our anticipated
    use case.
    Args:
        feature_names (:obj: `list` of :obj: `str`): List of names of columns to transform
        target_name (str): Name of the target value column
        **kwargs: Arbitrary dictionary of keyword arguments
    """

    def __init__(self, feature_names, target_name="approved", **_):
        self._feature_names = feature_names
        self._target_name = target_name
        self._risk_weight_lookup_dicts = dict()

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Calculate risk-weights for all values of each categorical variable in the training set.
        Risk weights are stored in the private `_risk_weight_lookup_dicts` class variable.
        This contains a dictionary for each categorical variable/column.  Each dictionary
        contains the risk-weight for all of that variable's distinct values in the training
        set.  It also contains the default risk-weight for novel values for each categorical
        variable.
        Args:
            X (pandas.Dataframe): Training data
        Returns:
            (object): self
        """
        for f in self._feature_names:
            tmp_df = X[[f, self._target_name]]
            tmp = (
                tmp_df.groupby(f).agg(lambda x: (x).sum() / x.count()).rename(columns={self._target_name: f}).to_dict()
            )
            tmp[f]["unknown"] = np.array(list(tmp[f].values())).mean()
            tmp[f][None] = np.array(list(tmp[f].values())).mean()

            self._risk_weight_lookup_dicts = {**self._risk_weight_lookup_dicts, **tmp}

        return self

    def transform(self, X):
        """Replaces columns of categorical values with their associated risk-weights.
        Args:
            X (pandas.Dataframe): Training data with columns of categorical variables.
        Returns:
            (numpy.array): Risk-weight values for each column of categoric input.
        """

        def default_to_unknown(x, dict_resource):
            return dict_resource.get(x, dict_resource["unknown"])

        tmp_arr = []
        for feature_name in self._feature_names:
            lookup_dict = self._risk_weight_lookup_dicts[feature_name]
            tmp = X[feature_name].apply(default_to_unknown, args=(lookup_dict,)).values
            tmp_arr.append(tmp)

        return np.array(tmp_arr).T
