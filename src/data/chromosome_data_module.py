from src.data.abstract_data_module import CrossValidationDataModule
from sklearn.model_selection import StratifiedKFold


class ChromosomeStratifiedDataModule(CrossValidationDataModule):
    """
    CrossValidationDataModule with chromosome-wise stratification.
    """

    def _create_folds(self):
        """Create fold assignments using chromosome stratification."""
        self.summary["fold"] = -1

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(
            skf.split(
                X=self.summary["gene"],
                y=self.summary["chromosome"],
            )
        ):
            self.summary.loc[test_idx, "fold"] = fold

        print("\nChromosome distribution by fold:")
        print(self.summary.groupby(["fold", "chromosome"]).size().unstack(fill_value=0))
