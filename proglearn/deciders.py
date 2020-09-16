'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from .base import BaseDecider, ClassificationDecider

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import type_of_target


class SimpleAverage(ClassificationDecider):
    """
    A class for a decider that uses the average vote for classification. 
    Uses ClassificationDecider as a base class.

    Parameters:
    -----------
    classes : list, default=[]
        Defaults to an empty list of classes.

    Attributes (class):
    -----------
    None

    Attributes (objects):
    -----------
    classes : list, default=[]
        Defaults to an empty list of classes.
    
    transformer_id_to_transformers : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary 
        maps transformers to a particular transformer id.
        
    transformer_id_to_voters : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.
    """

    def __init__(self, classes=[]):
        self.classes = classes

    def fit(
        """
        Function for fitting.
        Stores attributes (classes, transformer_id_to_transformers, 
        and transformer_id_to_voters) of a ClassificationDecider.
        
        Parameters:
        -----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
            
        transformer_id_to_transformers : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a transformer. This dictionary 
            maps transformers to a particular transformer id.
            
        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary thus
            maps voter classes to a particular transformer id.
            
        classes : list, default=None
            A list of classes of type obj.
            
        Returns:
        ----------
        SimpleAverage obj
            The ClassificationDecider object of class SimpleAverage is returned.
        """
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
        classes=None,
    ):
        self.classes = self.classes if len(self.classes) > 0 else np.unique(y)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        return self
    
    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities per input example.
        
        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Gets the mean vote per bag and append it to a vote per transformer id.
        Returns the average vote per transformer id.
        
        Parameters:
        -----------
        X : ndarray
            Input data matrix.
            
        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.
        
        Returns:
        -----------
        Returns mean vote across transformer ids.
        """
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.vote(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class.
        
        Uses the predict_proba method to get the mean vote per id. 
        Returns the class with the highest vote.
        
        Parameters:
        -----------
        X : ndarray
            Input data matrix.
            
        transformer_ids : list?, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.
            
        Returns:
        -----------
        An ndarray of type int where n is the length of the input data matrix X.
        """
        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        return self.classes[np.argmax(vote_overall, axis=1)]


class KNNRegressionDecider(BaseDecider):
    """
    A class for a decider that uses the k nearest-neighbors for regression. 
    Uses BaseDecider as a base class.

    Parameters:
    -----------
    classes : k, default=None
        Number of nearest neighbors to consider. Defaults to None.

    Attributes (class):
    -----------
    None

    Attributes (objects):
    -----------
    k : dict, default=None
        Number of nearest neighbors to consider. Defaults to None.
        
    is_fitted : boolean
        A boolean value for whether the decider is fitted.
    """

    def __init__(self, k=None):
        self.k = k
        self._is_fitted = False

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
        """
        Function for fitting.
        Stores attributes (classes, transformer_id_to_transformers, 
        and transformer_id_to_voters) of a BaseDecider.
        
        Parameters:
        -----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
            
        transformer_id_to_transformers : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a transformer. This dictionary 
            maps transformers to a particular transformer id.
            
        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary thus
            maps voter classes to a particular transformer id.
            
        Returns:
        ----------
        KNNRegressionDecider obj
            The BaseDecider object of class KNNRegressionDecider is returned.
        """
        X, y = check_X_y(X, y)
        n = len(y)
        if not self.k:
            self.k = min(16 * int(np.log2(n)), int(0.33 * n))

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        yhats = self.ensemble_represetations(X)

        self.knn = KNeighborsRegressor(self.k, weights="distance", p=1)
        self.knn.fit(yhats, y)

        self._is_fitted = True
        return self

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        yhats = self.ensemble_represetations(X)

        return self.knn.predict(yhats)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        n = len(X)

        # transformer_ids input is ignored - you can only use
        # the transformers you are trained on.
        yhats = np.zeros((n, len(self.transformer_ids)))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for i in range(len(self.transformer_ids)):

            transformer_id = self.transformer_ids[i]

            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, i] = voter.vote(X_transformed).reshape(n)

        return yhats


class LinearRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_
        self._is_fitted = False

    def fit(
        self, X, y, transformer_id_to_transformers, transformer_id_to_voters,
    ):
        X, y = check_X_y(X, y)

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        yhats = self.ensemble_represetations(X)

        self.ridge = Ridge(self.lambda_)
        self.ridge.fit(yhats, y)

        self._is_fitted = True
        return self

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        yhats = self.ensemble_represetations(X)

        return self.ridge.predict(yhats)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        n = len(X)

        # transformer_ids input is ignored - you can only use the transformers
        # you are trained on.
        yhats = np.zeros((n, len(self.transformer_ids)))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for i in range(len(self.transformer_ids)):

            transformer_id = self.transformer_ids[i]

            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, i] = voter.vote(X_transformed).reshape(n)

        return yhats