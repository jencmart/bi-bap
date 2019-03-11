from abc import ABC, abstractmethod


class AbstractRegressor(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 10,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True):
        raise NotImplementedError("You must implement this")

