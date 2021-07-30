import os
import argparse
import optparse
import numpy as np
import matplotlib as plt
import scipy as sp
import scipy.constants
from scipy.optimize import minimize, differential_evolution, Bounds, NonlinearConstraint
from scipy.misc import derivative
from sklearn.metrics import mean_squared_error
from subprocess import call

from string import Template
import pandas as pd
import matplotlib.pyplot as plt


# ICOOL_TEMP = "/afs/cern.ch/user/e/efol/icool/run/for001_template.txt"
# ICOOL_PATH = "/afs/cern.ch/user/e/efol/icool/run"

ICOOL_TEMP = "./for001_template.txt"
ICOOL_PATH = "./"

# muon mass in GeV
mu_mass = scipy.constants.physical_constants['muon mass energy equivalent in MeV'][0] * 0.001
c = scipy.constants.c


plt.rcParams.update({"font.size":18, 'figure.figsize':(10, 5), "legend.fancybox" : True})

optimized_vars = []
opt_loss = []

# TODO: transmission!


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
    nfinal = 0
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
            elif parts[0].startswith("particles"):
                nfinal = parts[len(parts)-1]
    dataframe = pd.DataFrame(table_rows, columns=column_names)
    return dataframe, nfinal




def get_bz(output_file):
    table_rows = []
    column_names = []
    output = open(output_file, "r")
    output_lines = output.readlines()
    for line in output_lines[2:]:
        parts = line.split()
        parts = [part.strip() for part in parts]
        if line.startswith("#"):
            column_names.append('flg')
            column_names.append('z')
            column_names.extend(parts[15:])
        else:
            row = [parts[3]] + [parts[8]] + parts[14:]
            table_rows.append(np.array(row, dtype=float))
    df = pd.DataFrame(table_rows, columns=column_names)
    bz = df.loc[df['flg'] == 0, 'Bz'].values
    z = df.loc[df['flg'] == 0, 'z'].values
    return bz, z




def plot_from_icool(output_file):
    bz, z = get_bz(output_file)

    plt.plot(z, bz) #, label="clen=0.4, elen=0.05, alen=0.02")
    plt.xlabel("z [m]")
    plt.ylabel("B(z)[T]")

    # plt.legend()
    plt.tight_layout()
    plt.show()




def plot_from_ecalc(df, nfinal, ninit):
    """
    Plotting beam properties depending on provided keywords.
    "emittance_trans_long": plots e_transv and e_long with 2 y-axis
    "emitance_and_beta": plots e_transc and beta function with 2 y-axis
    "emittance_and_pz": plots change in emit_trans and beam momentum
    "all": plotting of all beam parameters
    """
    print("Transmission: {}%".format(float(nfinal)/float(ninit) * 100))
    plot_beta_alpha(df)
    plot_emittances(df)
    # plot_emit_beta(df)
    plot_pz(df)
    

def plot_emittances(df):
    # handling of start of the cell, longitudinal emittance
    df["elong"].values[df["elong"].values < 0] = 0
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel(r"$\varepsilon_{t} [\mu m]$", color=color)
    ax1.plot(df.Z, df.eperp*1e6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\varepsilon_{l}$ [mm]", color=color)
    ax2.plot(df.Z, df.elong*1e3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    # plt.vlines([2, 2.5], np.min(df.elong*1e3), np.max(df.elong*1e3), linestyles="--")
    plt.show()
    # plt.savefig("emit.pdf")


def plot_emit_beta(df):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel(r"$\varepsilon_{t} [\mu m]$", color=color)
    ax1.plot(df.Z, df.eperp*1e6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\beta$ [cm]", color=color)
    ax2.plot(df.Z, df.beta*1e2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.vlines([2, 2.5], np.min(df.beta*1e3), np.max(df.beta*1e3), linestyles="--")
    plt.show()


def plot_pz(df):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel(r"$P_{z} [GeV]$")
    ax1.plot(df.Z, df.Pzavg)
    fig.tight_layout()
    plt.vlines([1.98, 2.48], np.min(df.Pzavg), np.max(df.Pzavg), linestyles="--")
    plt.show()



def plot_beta_alpha(df):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel(r"$\beta$ [cm]", color=color)
    ax1.plot(df.Z, df.beta*1e2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color='tab:blue'
    ax2.set_ylabel(r"$\alpha$", color=color)
    ax2.plot(df.Z, df.alpha, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    # plt.vlines([2, 2.5], np.min(df.alpha), np.max(df.alpha),linestyles="--")
    plt.show()
    # plt.savefig("beta_alpha.pdf")

    plt.plot(df.Z, df.loc[:,"Lcan(eVs)"], label=r"$L_{can}$")
    plt.vlines([1.98, 2.48], np.min(df.loc[:,"Lcan(eVs)"]), np.max(df.loc[:,"Lcan(eVs)"]), linestyles="--")
    plt.legend()
    # plt.savefig("lcan.pdf")
    plt.show()



def run_ecalc():
    # run ecalc with defaults
    call("./ecalc9")
    dataframe, nfinal = create_data_frame("ecalc9.out")
    return dataframe, nfinal


        
def run_icool(param):
    if param is not None:
        with open("./for001_optimizer_template.dat", 'r') as template:
            template_str = template.read()
        t = Template(template_str)
        lf_slen = 1.98
        hf_slen = 0.5

        elen_lf_1 = param[0]
        att_lf_1 = param[1]
        elen_hf = param[2]
        att_hf = param[3]
        elen_lf_2 = param[4]
        att_lf_2 = param[5]

        start_n_reg = int(elen_hf / 0.0125)
        center_n_reg = int(40-2*start_n_reg)
        if 2*start_n_reg + center_n_reg == 40:
            end_n_reg = start_n_reg
        elif center_n_reg == 40:
            center_n_reg = 38
            start_n_reg = 1
            end_n_reg = 1
        else:
            end_n_reg = 40 - start_n_reg - center_n_reg
        absorber = "VAC"

        # content_to_run = t.substitute(
        #         CLEN_LF_1 = str(lf_slen-2*elen_lf_1), ELEN_LF_1 = str(elen_lf_1), ATT_LF_1 = str(att_lf_1),
        #                                 CLEN_HF = str(hf_slen-2*elen_hf), ELEN_HF = str(elen_hf), ATT_HF = str(att_hf),
        #                                 CLEN_LF_2 = str(lf_slen-2*elen_lf_2), ELEN_LF_2 = str(elen_lf_2), ATT_LF_2 = str(att_lf_2)
        #                                 )
        content_to_run = t.substitute(
                CLEN_LF_1 = str(lf_slen-2*elen_lf_1), ELEN_LF_1 = str(elen_lf_1), ATT_LF_1 = str(att_lf_1),
                                        CLEN_HF = str(hf_slen-2*elen_hf), ELEN_HF = str(elen_hf), ATT_HF = str(att_hf),
                                        HF_START_N_REG = str(start_n_reg), CENTER_N_REG = str(center_n_reg), HF_END_N_REG = str(end_n_reg), ABSORBER = absorber,
                                        CLEN_LF_2 = str(lf_slen-2*elen_lf_2), ELEN_LF_2 = str(elen_lf_2), ATT_LF_2 = str(att_lf_2)
                                        )
        _write_command_file(ICOOL_PATH, content_to_run)
    call("../icool")



def _read_template():
    with open(ICOOL_TEMP, 'r') as template:
        template_str = template.read()
    return template_str


def resolve_template(template_str, emit_t, pz_init, n_init, sol_l, n_sol,
                    b_sol, absorber):
    cell_l = sol_l * n_sol
   
    sigma_xy, sigma_px_py, beta = define_beam(emit_t, pz_init, b_sol)
    t = Template(template_str)
    content_to_run = t.substitute(SIGMA_XY=str(sigma_xy), SIGMA_PXY=str(sigma_px_py), 
        PZ=str(pz_init), SOL_L=str(sol_l), N_SOL=str(n_sol), N_PART=str(n_init), 
        CELL_L=str(cell_l), HF_SOL_B=str(b_sol), ABSORB=str(absorber))
    return content_to_run


def _write_command_file(ICOOL_PATH, content):
    file_path = os.path.join(ICOOL_PATH, "for001.dat")
    with open(file_path, "w") as f:
        f.write(content)


def define_beam(emit_n, pz_init, b_sol):
    beta_init = ((2*pz_init)/(0.3*b_sol))
    sigma_xy = np.sqrt((2*emit_n*mu_mass) / (0.3*b_sol))
    sigma_xy_prime = np.sqrt(((emit_n * mu_mass)/pz_init)*((0.3*b_sol)/(2*pz_init)))
    sigma_px_py = pz_init*sigma_xy_prime  

    print("For transverse emittance {0} [m], pz {1} [GeV], solenoid with {2} [T]:".format(emit_n, pz_init, b_sol))
    print("beta = {0} [m]".format(beta_init))
    print("Sigma X, Sigma Y: {}".format(sigma_xy))
    print("Sigma Px, Py = {}".format(sigma_px_py))
    return sigma_xy, sigma_px_py, beta_init
    


# Function that normalizes paramters
def p_normalize(p, p_ave, p_diff):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm


# Function that un-normalizes parameters
def p_un_normalize(p, p_ave, p_diff):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm


def optimization_step(p):
    beta_lf = 0.257 # 0.01904 for p = 0.01, B=3.5 # 0.257 (p=0.135, B = 3.5)
    beta_hf = 0.03 # 0.0133 p=0.01, emitt = 10, B=5T # 0.18 (p=0.135, B = 5T) # 0.03 (0.135, 35T)
    run_icool(p)
    df, nfinal = run_ecalc()
    bz, z = get_bz("for009.dat")
    emit = df.eperp

    first_lf_start_idx = df.loc[df['Z'] == 0.0E+01].index.tolist()[0]
    first_lf_end_idx = df.loc[df['Z'] == 0.1980E+01].index.tolist()[0]
    first_lf_beta_icool = df.beta[first_lf_start_idx:first_lf_end_idx]
    
    hf_start_idx = df.loc[df['Z'] == 0.1992E+01].index.tolist()[0] # 0.2000E+01].index.tolist()[0]
    hf_end_idx = df.loc[df['Z'] == 0.2480E+01].index.tolist()[0]
    hf_beta_icool = df.beta[hf_start_idx:hf_end_idx]
  
    sec_lf_start_idx = df.loc[df['Z'] == 0.2546E+01].index.tolist()[0] # 0.3046E+01].index.tolist()[0]
    sec_lf_end_idx = df.loc[df['Z'] == 0.446E+01].index.tolist()[0] # 0.496E+01].index.tolist()[0]
    sec_lf_beta_icool = df.beta[sec_lf_start_idx:sec_lf_end_idx]

    lf_1_center = df.loc[df['Z'] == 0.9900E+00, 'beta'].values[0]
    hf_center = df.loc[df['Z'] == 0.2305E+01, 'beta'].values[0]
    lf_2_center = df.loc[df['Z'] == 0.3470E+01, 'beta'].values[0]


    loss =  10*np.mean(np.abs(df.alpha[hf_start_idx:hf_end_idx])) + 5*np.abs(hf_center - beta_hf) + \
                5*np.abs(df.beta[sec_lf_start_idx]-beta_lf) + 5*np.abs(lf_2_center-beta_lf) + 5*np.abs(df.beta[sec_lf_end_idx] - beta_lf) + \
                20 * np.mean(np.abs(df.alpha[sec_lf_start_idx:sec_lf_end_idx]))

    return loss


# This function defines one step of the ES algorithm at iteration i
def ES_step(nES, dtES, wES, kES, aES, p_n, i, cES_now, amplitude):
    
    # ES step for each parameter
    p_next = np.zeros(nES)
    
    # Loop through each parameter
    for j in np.arange(nES):
        p_next[j] = p_n[j] + amplitude*dtES*np.cos(dtES*i*wES[j]+kES*cES_now)*(aES[j]*wES[j])**0.5
    
        # For each new ES value, check that we stay within min/max constraints
        if p_next[j] < -1.0:
            p_next[j] = -1.0
        if p_next[j] > 1.0:
            p_next[j] = 1.0
            
    # Return the next value
    return p_next


def es_optimization(ES_steps, init_params, p_min, p_max, osc_size, k_es, plot_on):
    # Number of parameters being tuned
    nES = len(p_min)

    # Average values for normalization
    p_ave = (p_max + p_min)/2.0

    # Difference for normalization
    p_diff = p_max - p_min

    # This keeps track of the history of all of the parameters being tuned
    pES = np.zeros([ES_steps,nES])

    # Start with initial conditions inside of the max/min bounds
    # In this case I will start them near the center of the range
    # pES[0] = p_ave
    pES[0] = init_params

    # This keeps track of the history of all of the normalized parameters being tuned
    pES_n = np.zeros([ES_steps,nES])

    # Calculate the mean value of the initial condtions
    pES_n[0] = p_normalize(pES[0], p_ave, p_diff)

    # This keeps track of the history of the measured cost function
    cES = np.zeros(ES_steps)

    # Calculate the initial cost function value based on initial conditions
    cES[0] = optimization_step(pES[0])

    # ES dithering frequencies
    wES = np.linspace(1.0,1.75,nES)

    # ES dt step size
    dtES = 2*np.pi/(10*np.max(wES))


    # The values of aES and kES will be different for each system, depending on the
    # detailed shape of the functions involved, an intuitive way to set these ES
    # parameters is as follows:
    # Step 1: Set kES = 0 so that the parameters only oscillate about their initial points
    # Step 2: Slowly increase aES until parameter oscillations are big enough to cause
    # measurable changes in the noisy function that is to be minimized or maximized
    # Step 3: Once the oscillation amplitudes, aES, are sufficiently big, slowly increase
    # the feedback gain kES until the system starts to respond. Making kES too big
    # can destabilize the system


    # ES dithering size
    oscillation_size = osc_size
    aES = wES*(oscillation_size)**2
    # Note that each parameter has its own frequency and its own oscillation size

    # ES feedback gain kES (set kES<0 for maximization instead of minimization)
    kES = k_es

    # If you want the parameters to persistently oscillate without decay, set decay_rate = 1.0
    decay_rate = 1.0

    # Decay amplitude (this gets updated by the decay_rate to lower oscillation sizes
    amplitude = 1.0


    # Now we start the ES loop
    for i in np.arange(ES_steps-1):
        
        # Normalize previous parameter values
        # pES_n[i] = p_normalize(pES[i], p_ave, p_diff)
        
        # Take one ES step based on previous cost value
        pES_n[i+1] = ES_step(nES, dtES, wES, kES, aES, pES_n[i], i, cES[i], amplitude)
        
        # Un-normalize to physical parameter values
        pES[i+1] = p_un_normalize(pES_n[i+1], p_ave, p_diff)
        
        # Calculate new cost function values based on new settings
        cES[i+1] = optimization_step(pES[i+1])
        
        # Decay the amplitude
        amplitude = amplitude*decay_rate

    print("Optimized parameters: {}".format(pES))
    if plot_on:
        # Plot some results
        plt.figure(2,figsize=(10,15))
        plt.subplot(2,1,1)
        # plt.title(f'$k_{{ES}}$={kES}, $a_{{ES}}$={aES}')
        plt.plot(cES)
        plt.ylabel('ES cost')
        plt.xticks([])


        plt.subplot(2,1,2)
        plt.plot(pES_n[:,0],label='$p_{ES,1,n}$')
        plt.plot(pES_n[:,1],label='$p_{ES,2,n}$')
        plt.plot(pES_n[:,2],label='$p_{ES,1,n}$')
        plt.plot(pES_n[:,3],label='$p_{ES,2,n}$')
        plt.plot(pES_n[:,4],label='$p_{ES,1,n}$')
        plt.plot(pES_n[:,5],label='$p_{ES,2,n}$')
        plt.plot(1.0+0.0*pES_n[:,0],'r--',label='bounds')
        plt.plot(-1.0+0.0*pES_n[:,0],'r--')
        plt.legend(frameon=False)
        plt.ylabel('Normalized Parameters')
        plt.xlabel('ES step')

        plt.tight_layout()
        plt.show()
    return pES[len(pES)-1]



def param_scan(iter_n, init_params, bounds):
    # cons = {'type':'ineq', 'fun': lambda var : np.array([0.5 + var[0] + var[1],
    #                                                     0.25 + var[2] + var[3],
    #                                                     0.5 + var[4] + var[5]])}

    # res = minimize(optimization_step, init_params, method='SLSQP', bounds=bounds,
    #     options={'rhobeg': 0.7, 'maxiter': iter_n, 'disp': True})
    # can be parallized with workers = -1
    res = differential_evolution(optimization_step, bounds=bounds, popsize = 6, maxiter=iter_n)
    # res = minimize(optimization_step, init_params, method='Nelder-Mead', options={'maxiter': iter_n})
    result = res.x
    return res.x


def run_optimisation(method, matching, cooling):
    # parameters are end length and attenuation length of each solenoid iin the cell (3 solenoids = 6 variables)
    if matching:
        iter_n = 200
        bounds = [(0.001, 0.1), (0.001, 0.5), (0.05, 0.1),(0.075, 0.15), (0.001, 0.1), (0.001, 0.5)]
        params = [0.05, 0.25, 0.05, 0.1, 0.05, 0.25]
        if method == "ES":
            params = es_optimization(iter_n, params, np.array([bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0], bounds[4][0], bounds[5][0]]),
                np.array([bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1], bounds[4][1], bounds[5][1]]),
                osc_size = 0.15, k_es = 0.5, plot_on = True)
        elif method == "diff_evolution":
            params = param_scan(iter_n, params, bounds)
    #if cooling:

    return params


if __name__ == '__main__':
    # sigma_xy, sigma_px_py, beta_init = define_beam(emit_n=300*1e-06, pz_init=0.135, b_sol=3.5)
    opt_params = run_optimisation("diff_evolution", matching=True, cooling=False)
    print("Optimal results: {}".format(opt_params))
    # opt_params = None just runs exisiting for001 without replacing template settings
    # run_icool(param=[0.03374751, 0.00773362, 0.0611517, 0.14956967, 0.00195243, 0.00102642])
    dataframe, nfinal = run_ecalc()
    plot_from_ecalc(dataframe, nfinal, ninit=50)
    plot_from_icool("for009.dat")
    

# ES, 6 variables:  [0.03374751 0.00773362 0.0611517  0.14956967 0.00195243 0.00102642]


#From ES: [0.01258175 0.1069814]


# HF: [0.01687521 0.09998611]
#  HF:  [0.03491961 0.19618869], well matched, obj function: np.mean(np.abs(df.alpha)) - 0.01*bz[len(bz)/2] + np.sqrt(np.mean(np.diff(emit*1e4)**2)) +\
#            np.sqrt(np.mean((beta_lf - sec_lf_beta_icool)**2)) + np.sqrt(np.mean(np.diff(sec_lf_beta_icool)**2)) 

# Optmizing only HF, matched! 0.4734057192813917 0.013297140359304165 1 0.13729199076921184
# only HF, small oscillations in HF: [0.06469173 0.14366905]

   
# diff_evolution_rms: [0.01255352 0.01681555 0.08432478 0.10038093 0.01289015 0.01747886]
#2nd low field: 0.01050142 0.01807154

# absolute stable in 2nd LF:  [0.33859147 0.15450875 0.17778386 0.15156702 0.18737904 0.16792969]


# HF: 0.07070456 0.01039159 

# good result [0.01, 0.02, 0.1177, 0.1705, 0.0105, 0.01807]

# 5k optim. steps [2.00681038e-03 1.01630730e-04 9.46015915e-02 1.50357456e-01 6.79543021e-03 4.78802464e-03]

# best? [0.01255352, 0.01681555, 0.17778386, 0.15156702, 0.18737904, 0.16792969]