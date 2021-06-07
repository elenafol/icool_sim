import os
import argparse
import optparse
import numpy as np
import matplotlib as plt
import scipy as sp
import scipy.constants
from subprocess import call

from string import Template
import pandas as pd
import matplotlib.pyplot as plt


ICOOL_TEMP = "/afs/cern.ch/user/e/efol/icool/run/for001_template.txt"
ICOOL_PATH = "/afs/cern.ch/user/e/efol/icool/run"


plt.rcParams.update({"font.size":18, 'figure.figsize':(10, 5), "legend.fancybox" : True})


def _parse_args():
    # TODO: default values
    parser = optparse.OptionParser()
    parser.add_option("-e", "--emit_t", dest="emit_t",
                    help="Transverse emittance before entering the cooling cell, in [m].", type=float)
    parser.add_option("-p", "--pz_init", dest="pz_init",
                    help="Initial momentum pz, in [Gev/c].", type=float)
    parser.add_option("-n", "--n_init", dest="n_init",
                    help="Initial number of muons.", type=int)
    parser.add_option("-l", "--sol_l", dest="sol_l",
                    help="Legth of a single solenoid, in [m].", type=float)
    parser.add_option("-s", "--n_sol", dest="n_sol",
                    help="Number of high-field solenoids to be put in the cell.", type=int)
    parser.add_option("-b", "--b_sol", dest="b_sol",
                    help="Central field of high-field solenoids, in [T].", type=float)
    parser.add_option("-a", "--absorber", dest="absorber",
                    help="Absorber material, see icool manual.", type=str)
    (options, args) = parser.parse_args()
    return options.emit_t, options.pz_init, options.n_init, options.sol_l, options.n_sol, options.b_sol, options.absorber



def create_data_frame(ecalc_file):
    table_rows = []
    column_names = []
    with open(ecalc_file, "r") as data:
        for line in data:
            parts = line.split()
            parts = [part.strip() for part in parts]
            if len(parts) == 0:
                continue
            if line.startswith("regn"):
                for part in parts[2:]:
                    column_names.append(part)
            elif parts[0].isdigit():
                table_rows.append(np.array(parts[1:], dtype=float))
    dataframe = pd.DataFrame(table_rows, columns=column_names)
    return dataframe



def plot_from_ecalc():
    """
    Plotting beam properties depending on provided keywords.
    "emittance_trans_long": plots e_transv and e_long with 2 y-axis
    "emitance_and_beta": plots e_transc and beta function with 2 y-axis
    "emittance_and_pz": plots change in emit_trans and beam momentum
    "all": plotting of all beam parameters
    """
    df = create_data_frame("ecalc9.out")
    plot_emittances(df)
    plot_emit_beta(df)
    plot_pz(df)
    plot_beta_alpha(df)
    

def plot_emittances(df):
    # handling of start of the cell, longitudinal emittance
    df["elong"].values[df["elong"].values < 0] = 0
    
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("cell [m]")
    ax1.set_ylabel(r"$\varepsilon_{t} [\mu m]$", color=color)
    ax1.plot(df.Z, df.eperp*1e6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\varepsilon_{l}$ [mm]", color=color)
    ax2.plot(df.Z, df.elong*1e3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()


def plot_emit_beta(df):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("cell [m]")
    ax1.set_ylabel(r"$\varepsilon_{t} [\mu m]$", color=color)
    ax1.plot(df.Z, df.eperp*1e6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\beta$ [cm]", color=color)
    ax2.plot(df.Z, df.beta*1e3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()


def plot_pz(df):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("cell [m]")
    ax1.set_ylabel(r"$P_{z} [GeV]$")
    ax1.plot(df.Z, df.Pzavg)
    fig.tight_layout()
    plt.show()


def plot_beta_alpha(df):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("cell [m]")
    ax1.set_ylabel(r"$\beta$ [cm]", color=color)
    ax1.plot(df.Z, df.beta*1e3, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\alpha$", color=color)
    ax2.plot(df.Z, df.alpha, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()


def run_ecalc():
    call("./ecalc9")


def run_icool(emit_t, pz_init, n_init, sol_l, n_sol, b_sol, absorber):
    template_str = _read_template()
    content_to_run = resolve_template(template_str, emit_t, pz_init, n_init, sol_l, n_sol,
                    b_sol, absorber)
    _write_command_file(ICOOL_PATH, content_to_run)
    call("../icool")


def _read_template():
    with open(ICOOL_TEMP, 'r') as template:
        template_str = template.read()
    return template_str


def resolve_template(template_str, emit_t, pz_init, n_init, sol_l, n_sol,
                    b_sol, absorber):
    cell_l = sol_l * n_sol
    beta_init = ((2*pz_init)/(0.3*b_sol))
    rms_emit = np.sqrt(emit_t*beta_init)
    t = Template(template_str)
    content_to_run = t.substitute(RMS_EMIT=str(rms_emit), PZ=str(pz_init), SOL_L=str(sol_l), 
        N_SOL=str(n_sol), N_PART=str(n_init), CELL_L=str(cell_l), 
        HF_SOL_B=str(b_sol), ABSORB=str(absorber))
    return content_to_run


def _write_command_file(ICOOL_PATH, content):
    file_path = os.path.join(ICOOL_PATH, "for001.dat")
    with open(file_path, "w") as f:
        f.write(content)



if __name__ == '__main__':
    # run_icool(*_parse_args())
    # run_ecalc()
    plot_from_ecalc()