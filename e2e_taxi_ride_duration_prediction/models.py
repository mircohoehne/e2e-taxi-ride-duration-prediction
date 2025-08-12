from typing import Protocol, Self, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix


class SklearnCompatibleRegressor(Protocol):
    def fit(self, X: Union[spmatrix, np.ndarray], y: ArrayLike) -> Self: ...
    def predict(self, X: Union[spmatrix, np.ndarray]) -> ArrayLike: ...
