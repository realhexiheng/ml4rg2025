from src.data.abstract_data_module import CrossValidationDataModule, HDFReader


def test_datamodule():
    print("\n" + "=" * 50)
    print("Testing train/validation/test mode")
    print("=" * 50)

    datamodule = CrossValidationDataModule(
        reader=HDFReader("data/prepared/genewise.h5"),
        dataset="data/prepared/chromosome_stratified_dataset",
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

    print(f"Train batch embeddings shape: {train_batch['embeddings'].shape}")
    print(f"Train batch expressions shape: {train_batch['expressions'].shape}")

    val_loader = datamodule.val_dataloader()
    if val_loader is not None:
        print("Testing validation dataloader...")
        val_batch = next(iter(val_loader))
        print(f"Val batch embeddings shape: {val_batch['embeddings'].shape}")
        print(f"Val batch expressions shape: {val_batch['expressions'].shape}")
    else:
        print("No validation dataloader (validation=False)")

    print("Setting up test data...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))

    print(f"Test batch embeddings shape: {test_batch['embeddings'].shape}")
    print(f"Test batch expressions shape: {test_batch['expressions'].shape}")
    print("All tests passed.")


def test_datamodule_no_validation():
    print("\n" + "=" * 50)
    print("Testing train/test only mode (no validation)")
    print("=" * 50)

    datamodule = CrossValidationDataModule(
        reader=HDFReader("data/prepared/genewise.h5"),
        dataset="data/prepared/chromosome_stratified_dataset",
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

    print(f"Train batch embeddings shape: {train_batch['embeddings'].shape}")
    print(f"Train batch expressions shape: {train_batch['expressions'].shape}")

    # Should return None when validation=False
    val_loader = datamodule.val_dataloader()
    print(f"Validation loader: {val_loader} (should be None)")

    print("Setting up test data...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))

    print(f"Test batch embeddings shape: {test_batch['embeddings'].shape}")
    print(f"Test batch expressions shape: {test_batch['expressions'].shape}")
    print("Train/test only mode passed.")


def test_all():
    """Run all tests."""
    test_datamodule()
    test_datamodule_no_validation()
    print("All tests passed.")


if __name__ == "__main__":
    test_all()
