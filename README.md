# Parnassus-Core

## Installation

To install the `parnassus-core` package, ensure you have Python 3.10 or 3.11 installed. Then, run the following commands:

```bash
# Clone the repository
git clone https://github.com/dkobylianskii/parnassus-core/
cd parnassus

# Install package
pip install -e .
```

To additionally install the development dependencies (for formatting and linting) use
```bash
pip install -e '.[dev]'
```

## Development

You can set up and run pre-commit hooks with

```bash
pre-commit install
pre-commmit run --all-files
```

## Usage

Each experiment has its own directory under `src/parnassus_core/experiments/`.
For launch it should contain a `dataset.py` file that defines a dataset class and a `main.py` file that inherits `parnassus_core.flow.utils.cli.CLI` class.
You can run the main script of an experiment from the command line. For example, to run the CMS 2011 experiment, navigate to the `src/parnassus_core/experiments/cms_2011/` directory and run:

```bash
python main.py fit -c configs/event.yaml
```
For more details on the experiment, refer to the `README.md` file in the `cms_2011` directory.

## Evaluation
You can run the evaluation scripts directly from the command line. For example, to evaluate the CMS 2011 experiment, navigate to the `src/parnassus_core/experiments/cms_2011/evaluation/` directory and run:

```bash
python eval.py -c config.yaml
```

## Project Structure

This project is structured in the following way:

- `src/`
	- `parnassus-core/`
        - `experiments/`
            - `cms_2011/`
        - `evaluation/`
        - `preprocessing/`
        - `utils/`
        - `flow/`
            - `models/`
            - `utils/`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
