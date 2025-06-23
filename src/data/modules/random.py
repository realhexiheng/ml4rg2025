from src.data.modules._abstract import CrossValidationDataModule
from sklearn.model_selection import KFold


class RandomDataModule(CrossValidationDataModule):
    """
    CrossValidationDataModule with random gene splits (no stratification).
    """

    def _create_folds(self, n_folds: int):
        """Create fold assignments using random splits."""
        self.summary["fold"] = -1

        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(kf.split(self.summary["gene"])):
            self.summary.iloc[test_idx, self.summary.columns.get_loc("fold")] = fold
