from typing import Protocol, Self, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix


class SklearnCompatibleRegressor(Protocol):
    def fit(self, X: Union[csr_matrix, np.ndarray], y: ArrayLike) -> Self: ...
    def predict(self, X: Union[csr_matrix, np.ndarray]) -> ArrayLike: ...
