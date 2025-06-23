# ML4RG2025
project 06 - Predicting multiple RNA profiles at once on Saccharomyces cerevisiae

## Resources

- [Shared Drive](https://drive.google.com/drive/folders/1fq_oc8jSxdT0Eu7sk94D9UR4hftLevma?usp=sharing)
- [WandB](https://wandb.ai/nictru32-tum/RNA_prediction?nw=nwusernictru32)

## Data setup

1. Download the `genewise.h5` file from the "processed" folder of the shared drive.
2. Move the file to the `data/processed` folder.

## Environment Setup
This project uses [uv](https://pypi.org/project/uv/) for managing the development environment.

1. **Sync development environment**

    ```sh
    uv sync
    ```
    This will create a virtual environment `.venv` under project folder and install all the dependencies listed in the `pyproject.toml` file.

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
