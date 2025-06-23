from src.data.modules._abstract import CrossValidationDataModule
from sklearn.model_selection import StratifiedGroupKFold

class ParalogousGeneDataModule(CrossValidationDataModule):
    """
    CrossValidationDataModule with chromosome and paralog group stratification.
    """

    def _create_folds(self):
        """Create fold assignments using chromosome and paralog group stratification."""

        if "paralog_group" not in self.summary.columns:
            raise ValueError(
                "Summary CSV must contain 'paralog_group' column with paralog information. "
                "Please run preprocessing with include_paralogs=True."
            )

        self.summary["fold"] = -1

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(
            sgkf.split(
                self.summary["gene"],
                self.summary["chromosome"],
                groups=self.summary["paralog_group"],
            )
        ):
            self.summary.loc[test_idx, "fold"] = fold
