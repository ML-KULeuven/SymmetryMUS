# Exploiting symmetries in MUS Computation

This repository contains the code for the following paper:
> Bleukx, Ignace, et al. "Exploiting Symmetries in MUS Computation." (AAAI 2025).

> **Abstract**
> 
> In eXplainable Constraint Solving (XCS), it is common to
extract a Minimal Unsatisfiable Subset (MUS) from a set of
unsatisfiable constraints. This helps explain to a user why a
constraint specification does not admit a solution. Finding
MUSes can be computationally expensive for highly symmetric problems, as many combinations of constraints need
to be considered. In the traditional context of solving satisfaction problems, symmetry has been well studied, and effective
ways to detect and exploit symmetries during the search exist. However, in the setting of finding MUSes of unsatisfiable
constraint programs, symmetries are understudied. In this paper, we take inspiration from existing symmetry-handling
techniques and adapt well-known MUS-computation methods to exploit symmetries in the specification, speeding-up
overall computation time. Our results display a significant reduction of runtime for our adapted algorithms compared to
the baseline on symmetric problems.

## Overview of the repository
Algorithms:
- `deletion.py`: Python implementation of the deletion-based algorithm
- `ocus.py`: Python implementation of the implicit-hitting-set based algorithm for finding a smallest MUS
- `marco.py`: Python implementation of the MARCO-algorithm for enumerating MUSes/MCSes

Support-files:
- `breakid.py`: A Python-wrapper for BreakID
- `symmetries.py`: Houses the datastructures for storing generators
- `models.py`: Contains the functions to construct the bechmark-instances

## Running the code

To run  the code, you will need to install the CPMpy library and the BreakID tool.
**WARNING** make sure to install the BreakdID tool from the `pb_optimization` branch! 
https://bitbucket.org/krr/breakid/src/pb_optimization/

For efficient solving of symmetric problems, we also recommend to install the PB-solver Exact.

```bash
$ pip install cpmpy exact
```
BreakID can be downloaded and installed from [here](https://bitbucket.org/krr/breakid/src/master/)


## Citing
If you use this repository or ideas presented in the paper, please consider citing:
```bibtex
@inproceedings{bleukx2024exploiting,
  author       = {Ignace Bleukx and
                  H{\'{e}}l{\`{e}}ne Verhaeghe and
                  Bart Bogaerts and
                  Tias Guns},
  title        = {Exploiting Symmetries in {MUS} Computation},
  booktitle    = {{AAAI}},
  publisher    = {{AAAI} Press},
  year         = {2025}
}
```
