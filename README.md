# ML4RG2025
project 06 - Predicting multiple RNA profiles at once on Saccharomyces cerevisiae


# cross-tissue-learning
Repository to explore whether we can predict the state of a tissue based on another tissue from the same donor.

## Resoruces

[Project description](https://docs.google.com/document/d/1371zvQSwjMryL6-9ZU7JzNc-EwSsMsTFOqVXHBheF98/edit?tab=t.0)

[Data](https://drive.google.com/drive/u/0/folders/1IHKACrqhhqUHTQUjqecX9ttJ4w209gwN)

TODO - just markdown or google doc?

## Environment Setup
This project uses [uv](https://pypi.org/project/uv/) for managing the development environment.

    ```sh
    pip install uv
    ```

1. **Sync development environment**

    ```sh
    uv sync
    ```
    This will create a virtual environment `.venv` under project folder and install all the dependencies listed in the `pyproject.toml` file. `src` will be added with `setuptools` so that you can import the modules directly: `from preprocessing import preprocess_adata`.

    If you don't want to sync venv settings but prefer to have you own, you can run:
    
    ```sh
    uv init
    ```

    then install the dependencies as in next step.

2. **Install additional dependencies**
    For additional dependencies:
    ```sh
    uv install <package-name>
    ```
    Note `uv` currently support only packages from PyPI.

3. **Enter the virtual environment**

    ```sh
    source .venv/bin/activate
    ```

    and to exit the virtual environment, run:

    ```sh
    deactivate
    ```

## Baseline Model