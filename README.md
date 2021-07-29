# icool_sim


How to run icool from run_from_template.py

Required libraries: matplotlib, pandas, numpy

Options to be passed:

    -e: Transverse emittance before entering the cooling cell, in [m].
    -p: Initial momentum pz, in [Gev/c].
    -n: Number of particles to be tracked (initial).
    -l: Length of a single solenoid, in [m].
    -s: Number of high-field solenoids to be put in the cell.
    -b: Central field of high-field solenoids, in [T].
    -a: Absorber material, see icool manual.

command example: python run_from_template.py -e 0.0003 -p 0.1 -n 1000 -l 0.01 -s 50 -b 30.17 -a VAC


# Optimisation

The target is to match the optics in a cell consisting of a low field solenoid, high field solenoid and a second low field solenoid. In order to reduced the transverse emittance, an absorber will be placed into the peak field region of the high field solenoid. 

The optimisers are used to minimize an objective function defined for matching by varying the parameter of field model 2 (see SOL, model 2 in ICOOL manual). The parameters obtained in each optimisation steps are used to fill out for001_optimizer_template.dat. At each step icool runs with the created for001.dat file and output is analysed with ecalc. Using the ecalc output data, the new objective function is computed and optimisation continues.

Depending on the applied optimisation algorithm, the routine is finished when defined maximum number of iterations or another stopping criteria is reached. 
