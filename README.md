# Evolutionary study of decision making in the presence of social and personal norms
This repository contains the code for the agent-based simulations that supported the findings of the bachelor's thesis titled "Evolutionary study of decision making in the presence of social and personal norms", completed at Universidad Carlos III de Madrid (UC3M) during the 2023-2024 academic year.

## Dependencies
This implementation requires [MESA](https://mesa.readthedocs.io/en/stable/).

## Code Structure
The main script of this repository is `main.py`. The directory `utils` contains both `misc.py` and `mesa-model.py`. The file `misc.py` includes utility functions such as argument extraction and plotting, while `mesa-model.py` defines the model and agents used in the simulations. See section 3.2 in the thesis for more details on the code structure.

## Running the Code
To run the code and launch the simulation, it is required to pass, at least, the directory where images will be stored. Run `python main.py --help` for a detailed explanation of all the available parameters. Here are two examples of scripts to launch this code:

```bash
python main.py --game "Coor" --rounds 100 --pnb "uniform" --B "constant" --directory "/home/user/abm"
python main.py --game "CR" --rounds 1000 --pnb "normal" --B "constant" --G1 2.5 --G2 0.1 --directory "/home/user/abm"
```

## Running the Tests
To run the tests, simply launch, from the from the `utils` directory:

```bash
python unit_tests.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
