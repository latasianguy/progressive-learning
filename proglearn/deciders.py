"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .base import BaseClassificationDecider
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    NotFittedError,
)


class SimpleArgmaxAverage(BaseClassificationDecider):
    """
    A class for a decider that uses the average vote for classification.

    Parameters
    ----------
    classes : list, default=[]
        List of final output classification labels of type obj.

    Attributes
    ----------
    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.
    """

    def __init__(self, classes=[]):
        self.classes = classes

    def fit(
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
    ):
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

        Returns
        -------
        self : SimpleArgmaxAverage
            The object itself.

        Raises
        -------
        ValueError
            When the labels have not been provided and the classes are empty.
        """
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
        self.transformer_id_to_transformers_ = transformer_id_to_transformers
        self.transformer_id_to_voters_ = transformer_id_to_voters
        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities per input example.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Gets the mean vote per bagging component and append it to a vote per transformer id.
        Returns the aggregate average vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with specific transformer ids that will be used for inference. Defaults
            to using all transformers if no transformer ids are given.

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example


        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters_.keys()
        ):
            check_is_fitted(self)
            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers_[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers_[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters_[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example.

        Uses the predict_proba method to get the mean vote per id.
        Returns the class with the highest vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        return self.classes[np.argmax(vote_overall, axis=1)]

    

class KNNClassificationDecider(BaseClassificationDecider):
    """
    Doc string here.
    """

    def __init__(self, k=None, classes=[]):
        self.k = k
        self._is_fitted = False
        self.classes = classes

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
            
        self.k=int(np.log2(len(X)))
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
        
#         self.knn = KNeighborsClassifier(self.k, weights="distance", p=1)
#         self.knn.fit(yhats, y)

        self.knn = []
        temp_yhats = np.empty((len(yhats), len(self.transformer_ids)))
        for i in range(len(self.classes)):
            self.knn.append(KNeighborsClassifier(self.k, weights="distance", p=1))
            for j in range(len(self.transformer_ids)):
                temp_yhats[:, j] = yhats[:, len(self.classes)*j+i]
            class_label = np.array([1 if element == i else 0 for element in y])
            self.knn[i].fit(temp_yhats, class_label)

        self._is_fitted = True
        return self
    
    def predict_proba(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        X = check_array(X)
        
        yhats = self.ensemble_represetations(X)

        knn_out = np.empty((len(yhats), len(self.classes)))
        temp_yhats = np.empty((len(yhats), len(self.transformer_ids)))
        for i in range(len(self.classes)):
            for j in range(len(self.transformer_ids)):
                temp_yhats[:,j] = yhats[:, len(self.classes)*j+i]
            knn_out[:,i] = self.knn[i].predict_proba(temp_yhats)[:,1]
        
        normalized = knn_out/np.sum(knn_out, axis=1, keepdims=True)
        return normalized

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        knn_out = self.predict_proba(X)
        return np.argmax(knn_out, axis=1)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        n = len(X)
        c = len(self.classes)

        # transformer_ids input is ignored - you can only use
        # the transformers you are trained on.
        yhats = np.zeros((n, c * len(self.transformer_ids)))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for i in range(len(self.transformer_ids)):

            transformer_id = self.transformer_ids[i]

            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, i*c:i*c+c] = voter.predict_proba(X_transformed).reshape(n, c)

        return yhats