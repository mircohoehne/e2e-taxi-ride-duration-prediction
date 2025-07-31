from typing import Any, Protocol

from numpy.typing import ArrayLike


class SklearnCompatibleRegressor(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> ArrayLike: ...
