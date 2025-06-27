import pandas as pd
import pytest
from src.data.modules.chromosome import ChromosomeStratifiedDataModule
from src.utils.io import HDFReader


def test_datamodule():
    print("\n" + "=" * 50)
    print("Testing train/validation/test mode")
    print("=" * 50)

    datamodule = ChromosomeStratifiedDataModule(
        reader=HDFReader("data/prepared/genewise.h5"),
        summary=pd.read_csv("data/summary.csv"),
        test_fold=0,
        validation=True,
        batch_size=4,
        num_workers=0,
    )

    print("Setting up data...")
    datamodule.setup("fit")

    print("Testing train dataloader...")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))

    assert train_batch[0].shape[0] == 4, "Train batch size should be 4"
    assert train_batch[0].shape[1] == 500, (
        "Train batch embeddings should have 500 context window"
    )
    assert train_batch[0].shape[2] == 768, (
        "Train batch embeddings should have 768 dimensions"
    )
    print(f"Train batch embeddings shape: {train_batch[0].shape}")
    print(f"Train batch expressions shape: {train_batch[1].shape}")

    val_loader = datamodule.val_dataloader()
    if val_loader is not None:
        print("Testing validation dataloader...")
        val_batch = next(iter(val_loader))
        print(f"Val batch embeddings shape: {val_batch[0].shape}")
        print(f"Val batch expressions shape: {val_batch[1].shape}")
    else:
        print("No validation dataloader (validation=False)")

    print("Setting up test data...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))

    print(f"Test batch embeddings shape: {test_batch[0].shape}")
    print(f"Test batch expressions shape: {test_batch[1].shape}")
    print("All tests passed.")


def test_datamodule_no_validation():
    print("\n" + "=" * 50)
    print("Testing train/test only mode (no validation)")
    print("=" * 50)

    datamodule = ChromosomeStratifiedDataModule(
        reader=HDFReader("data/prepared/genewise.h5"),
        summary=pd.read_csv("data/summary.csv"),
        test_fold=0,
        validation=False,
        batch_size=4,
        num_workers=0,
    )

    print("Setting up data...")
    datamodule.setup("fit")

    print("Testing train dataloader...")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))

    assert train_batch[0].shape[0] == 4, "Train batch size should be 4"
    assert train_batch[0].shape[1] == 500, (
        "Train batch embeddings should have 500 context window"
    )
    assert train_batch[0].shape[2] == 768, (
        "Train batch embeddings should have 768 dimensions"
    )
    print(f"Train batch expressions shape: {train_batch[1].shape}")

    # Should return None when validation=False
    val_loader = datamodule.val_dataloader()
    print(f"Validation loader: {val_loader} (should be None)")

    print("Setting up test data...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))

    print(f"Test batch embeddings shape: {test_batch[0].shape}")
    print(f"Test batch expressions shape: {test_batch[1].shape}")
    print("Train/test only mode passed.")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
