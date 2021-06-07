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

command example: python run_from_template.py -e 0.0003 -p 0.1 -n 1000 -l 0.01 -s 50 -b 30.17 a VAC
