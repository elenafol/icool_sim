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
                for part in parts[1:]:
                    if part == '#':
                        part = "nreg"
                    column_names.append(part)
            elif parts[0].isdigit():
                table_rows.append(np.array(parts, dtype=float))
            elif parts[0].startswith("particles"):
                nfinal = parts[len(parts)-1]
    pd.set_option('display.precision', 6)
    dataframe = pd.DataFrame(table_rows, columns=column_names)
    return dataframe, nfinal



def df_from_icool(output_file):
    table_rows = []
    column_names = []
    output = open(output_file, "r")
    output_lines = output.readlines()
    for line in output_lines[2:]:
        parts = line.split()
        parts = [part.strip() for part in parts]
        if line.startswith("#"):
            column_names.append('evt')
            column_names.append('flg')
            column_names.append('reg')
            column_names.append('z')
            column_names.extend(parts[12:])
        else:
            row = [parts[0]] + [parts[3]] +[parts[4]] + [parts[8]] + parts[11:]
            table_rows.append(np.array(row, dtype=float))
    df = pd.DataFrame(table_rows, columns=column_names)
    return df



def plot_bz(output_file):
    df = df_from_icool(output_file)
    bz = df.loc[df['evt'] == 0, 'Bz'].values
    z = df.loc[df['evt'] == 0, 'z'].values
    plt.plot(z, bz) #, label="clen=0.4, elen=0.05, alen=0.02")
    plt.xlabel("z [m]")
    plt.ylabel("B(z)[T]")

    plt.tight_layout()
    plt.show()


# TODO: faster loop through dataframe
def get_energy_spread(output_file, plot):
    df = df_from_icool(output_file)
    regions = np.unique(df.reg.values)
    espread_in_reg = []
    ekin_in_reg = []
    z = []
    for reg in regions:
        flg_in_reg = df.loc[df['reg'] == reg, 'flg'].values
        if any(flg == 0 for flg in flg_in_reg):
            z.append(df.loc[df['reg'] == reg, 'z'].values[0])
            pz_in_reg = df.loc[(df['reg'] == reg) & (df['evt']!=0) & (df['flg'] == 0), 'Pz'].values
            ekin_part = []
            for pz in pz_in_reg:
                ekin_part.append(energy_mom_translation("pz", pz))
            espread_in_reg.append(np.std(ekin_part))
            ekin_in_reg.append(np.mean(ekin_part))
    if plot:
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("z [m]")
        ax1.set_ylabel(r'$\sigma E_{kin}$ [MeV]', color=color)
        ax1.plot(z, np.array(espread_in_reg)*1e3, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color="tab:blue"
        ax2.set_ylabel(r'$E_{kin}$ [MeV]', color=color)
        ax2.plot(z, np.array(ekin_in_reg)*1e3, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.show()

    return np.array(espread_in_reg)*1e3



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
    # plt.vlines([1.98, 2.48], np.min(df.Pzavg), np.max(df.Pzavg), linestyles="--")
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
    # plt.vlines([1.98, 2.48], np.min(df.loc[:,"Lcan(eVs)"]), np.max(df.loc[:,"Lcan(eVs)"]), linestyles="--")
    plt.legend()
    # plt.savefig("lcan.pdf")
    plt.show()



def run_ecalc():
    # run ecalc with defaults
    call("./ecalc9")
    df, nfinal = create_data_frame("ecalc9.out")
    emit_start = df.eperp.values[0] *1e6
    emit_end = df.eperp.values[len(df.eperp.values)-1]*1e6
    emit_red = ((emit_start-emit_end)/emit_start)
    print("Start emittance: {0}, Emittance at the end: {1}, Emittance reduction: {2} %".format(emit_start, emit_end, emit_red))
    return df, nfinal, emit_red

       
def run_icool(param):
    if param is not None:
        with open("./for001_optimizer_template.dat", 'r') as template:
            template_str = template.read()
        t = Template(template_str)
        lf_clen = 2
        hf_clen = 0.5

        elen_lf_1 = 0.001
        att_lf_1 = 0.001
        elen_hf = param[0]
        att_hf = param[1]
        elen_lf_2 =  param[2]
        att_lf_2 =  param[3]

        absorber = "VAC"

        # content_to_run = t.substitute(
        #         CLEN_LF_1 = str(lf_slen-2*elen_lf_1), ELEN_LF_1 = str(elen_lf_1), ATT_LF_1 = str(att_lf_1),
        #                                 CLEN_HF = str(hf_slen-2*elen_hf), ELEN_HF = str(elen_hf), ATT_HF = str(att_hf),
        #                                 CLEN_LF_2 = str(lf_slen-2*elen_lf_2), ELEN_LF_2 = str(elen_lf_2), ATT_LF_2 = str(att_lf_2)
        #                                 )
        content_to_run = t.substitute(ELEN_LF_1 = str(elen_lf_1), ATT_LF_1 = str(att_lf_1), LF1_END_LEN_REG = str(elen_lf_1 / 5),
                                        ELEN_HF = str(elen_hf), ATT_HF = str(att_hf), 
                                        HF_END_LEN_REG = str(elen_hf / 10), ABSORBER = absorber,
                                        ELEN_LF_2 = str(elen_lf_2), ATT_LF_2 = str(att_lf_2), LF2_END_LEN_REG = str(elen_lf_2 / 5)
                                        )
        _write_command_file(ICOOL_PATH, "for001.dat", content_to_run)
    call("../icool")


#abs_len is in meters
# TODO: replace in the template
def run_icool_sheet_mdl(sol_params, beam_params, absorber, abs_len):
    with open("./for018_tempplate_sheet_model", 'r') as template:
        template_str = template.read()
    t = Template(template_str)
    content_to_run = t.substitute(HF_RADIUS = str(sol_params[0]))
    _write_command_file(ICOOL_PATH, "for018.dat", content_to_run)
    with open("./for001_template_sheet_model", 'r') as template:
            template_str = template.read()
    t = Template(template_str)
    
    hf_region_absorber = int(abs_len /  0.01)
    hf_region_vac =  int((100 - hf_region_absorber) / 2)
    content_to_run = t.substitute(PZ = beam_params[0], SIGMAXY = beam_params[1], 
                        SIGMAPXY = beam_params[2], NINIT = beam_params[3], ABSORBER = absorber, VAC_REG=hf_region_vac, ABS_REG = hf_region_absorber)
    _write_command_file(ICOOL_PATH, "for001.dat", content_to_run)
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


def _write_command_file(ICOOL_PATH, file_name, content):
    file_path = os.path.join(ICOOL_PATH, file_name)
    with open(file_path, "w") as f:
        f.write(content)


def define_beam(emit_n, pz_init, b_lf, b_hf):
    # TODO: pz changes according to lower beta. How to consider this change?

    beta_lf = ((2*pz_init)/(0.3*b_lf))
    beta_hf = ((2*pz_init)/(0.3*b_hf))
    sigma_xy = np.sqrt((2*emit_n*mu_mass) / (0.3*b_lf))
    sigma_xy_prime = np.sqrt(((emit_n * mu_mass)/pz_init)*((0.3*b_lf)/(2*pz_init)))
    sigma_px_py = pz_init*sigma_xy_prime  

    print("For transverse emittance {0} [m], pz {1} [GeV], first solenoid with {2} [T]:".format(emit_n, pz_init, b_lf))
    print("beta LF = {0}, beta HF = {1}  [m]".format(beta_lf, beta_hf))
    print("Sigma X, Sigma Y: {}".format(sigma_xy))
    print("Sigma Px, Py = {}".format(sigma_px_py))
    return sigma_xy, sigma_px_py, beta_lf, beta_hf



# quantity argument is the quantity which is provided
def energy_mom_translation(quantity, value):
    res = value
    if quantity=="ekin":
        gamma = 1 + value/mu_mass
        res = mu_mass*np.sqrt(gamma**2-1)
    if quantity=="pz":
        gamma=np.sqrt((value/mu_mass)**2+1)
        res = mu_mass*(gamma-1)
    return res



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
    beta_hf = 0.03 # 0.0133 p=0.01, emitt = 10, B=5T # 0.18 (p=0.135, B = 5T) # 0.03 (0.135, 30T)
    run_icool(p)
    df, nfinal, emit_red = run_ecalc()

    # Filtering of beta values is based on ecalc output for 100 regions 
    # (max possible number of regions) and output locations defined in for001 template
    first_lf_beta_icool = df.beta[5:26]
    
    hf_beta_icool = df.beta[41:60]

    sec_lf_beta_icool = df.beta[76:97]

    lf_1_center = df.loc[df['nreg']==30, 'beta'].values[0]
    hf_center = df.loc[df['nreg']==100, 'beta'].values[0]
    lf_2_center = df.loc[df['nreg']==170, 'beta'].values[0]

    # print(np.abs(hf_center-beta_hf))
    # print(np.mean(np.abs(hf_beta_icool-beta_hf)))
    # print(np.abs(lf_2_center-beta_lf))
    # print(np.mean(np.abs(sec_lf_beta_icool-beta_lf)))
    # print("B-Feld difference to predefined 30T: {}".format(30-bz[(len(bz)-1)/2]))
    

    loss = 10*np.abs(hf_center-beta_hf) + 10*np.std(hf_beta_icool- beta_hf) + \
            np.abs(lf_2_center-beta_lf) + np.std(sec_lf_beta_icool -beta_lf) 

 # 10*np.mean(np.abs(hf_beta_icool-beta_hf)) + np.mean(np.abs(sec_lf_beta_icool-beta_lf)) 
    return loss



def optimize_radius(p, beam, beta_lf, beta_hf, absorber, abs_len):
    run_icool_sheet_mdl(p, beam, absorber, abs_len)
    df, nfinal, emit_red = run_ecalc()

    beta_center_hf = df.loc[df['Z'] == 2.5, 'beta'].values[0]
    beta_start = df.loc[df['Z'] == 0.01, 'beta'].values[0]
    beta_start_lf = df.loc[df['Z'] == 3.01, 'beta'].values[0]
    beta_end = df.loc[df['Z'] == 4.9, 'beta'].values[0]

    # bz_center = df.loc[df['Z'] == 2.5, 'Bz'].values[0]
    # bz_start = df.loc[df['Z'] == 0.01, 'Bz'].values[0]
    # bz_end = df.loc[df['Z'] == 4.9, 'Bz'].values[0]


    loss = np.abs(beta_center_hf - beta_hf)
    opt_loss.append(loss)
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
        # plt.plot(pES_n[:,4],label='$p_{ES,1,n}$')
        # plt.plot(pES_n[:,5],label='$p_{ES,2,n}$')
        plt.plot(1.0+0.0*pES_n[:,0],'r--',label='bounds')
        plt.plot(-1.0+0.0*pES_n[:,0],'r--')
        plt.legend(frameon=False)
        plt.ylabel('Normalized Parameters')
        plt.xlabel('ES step')

        plt.tight_layout()
        plt.show()
    return pES[len(pES)-1]



def param_scan(beam, beta_lf, beta_hf, absorber, abs_len, iter_n, init_params, bounds):
    # can be parallized with workers = -1
    res = differential_evolution(optimize_radius, args=(beam, beta_lf, beta_hf, absorber, abs_len), 
                bounds=bounds, popsize = 6, maxiter=iter_n)
    result = res.x
    return res.x


def run_optimisation(beam, beta_lf, beta_hf, absorber, method, matching, cooling):
    # parameters are end length and attenuation length of each solenoid iin the cell (3 solenoids = 6 variables)
    if matching:
        iter_n = 5
        bounds = [(0.3, 0.75)]
        params = [0.4]
        if method == "ES":
            params = es_optimization(iter_n, params, np.array([bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]]), #, bounds[4][0], bounds[5][0]]),
                np.array([bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]]), #, bounds[4][1], bounds[5][1]]),
                osc_size = 0.15, k_es = 0.5, plot_on = True)
        elif method == "diff_evolution":
            params = param_scan(beam, beta_lf, beta_hf, absorber, abs_len, iter_n, params, bounds)
    #if cooling:

    return params


def vary_ekin_plot():
    ninit=1000
    energy_scale = [10, 20, 30, 40, 50, 60, 70]     # Ekin in MeV
    absorbers = ["LH", "LHE", "LI"]
    emitt_red_in_absorbers = dict.fromkeys(absorbers, [])
    e_spread_in_absorbers = dict.fromkeys(absorbers, [])
    transmis_in_absorbers = dict.fromkeys(absorbers, [])
    
    for absorber in absorbers:
        emitt_red_in_absorbers[absorber] = []
        e_spread_in_absorbers[absorber] = []
        transmis_in_absorbers[absorber] = []
        for ekin in energy_scale:
            pz =  energy_mom_translation("ekin", ekin*0.001)  # icool unit is GeV
            sigma_xy, sigma_px_py, beta_lf, beta_hf = define_beam(emit_n=300*1e-06, pz_init=pz, b_lf=4.5, b_hf=30)
            beam = [pz, sigma_xy, sigma_px_py, ninit]
            run_icool_sheet_mdl(sol_params=[0.350089], beam_params = beam, absorber = absorber, abs_len=0.2)
            energy_spread_in_regions = get_energy_spread("for009.dat", plot=False)
            dataframe, nfinal, emit_red = run_ecalc()

            emitt_red_in_absorbers[absorber].append(emit_red)
            transmis_in_absorbers[absorber].append(float(nfinal)/float(ninit)*100)
            e_spread_in_absorbers[absorber].append(np.max(energy_spread_in_regions))

        plt.plot(np.array(transmis_in_absorbers.get(absorber)), np.array(energy_scale), label = absorber)
    print(transmis_in_absorbers)
    # plt.xlabel(r'$\Delta \varepsilon / \varepsilon_{init}$ [%]')
    # plt.xlabel('Transmission [%]')
    plt.xlabel(r'$\sigma E_{max}$ [MeV]')
    plt.ylabel(r'$E_{init}$ [MeV]')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ninit=1000
    pz = 0.135
    ekin = energy_mom_translation("pz", pz)
    print("Ekin for pz = {0} GeV/c is {1} GeV:".format(pz, ekin))
    absorber = "VAC"
    sigma_xy, sigma_px_py, beta_lf, beta_hf = define_beam(emit_n=300*1e-06, pz_init=pz, b_lf=4.5, b_hf=30)
    beam = [pz, sigma_xy, sigma_px_py, ninit]
    opt_params = run_optimisation(beam, beta_lf, beta_hf, absorber, abs_len, "diff_evolution", matching=True, cooling=False)
    opt_params = None just runs exisiting for001 without replacing template settings
    print("Optimal results: {}".format(opt_params))
    plt.plot(range(len(opt_loss)), opt_loss)
    plt.show()
    
    # absorbers = LH, LHE,  LIH, LI, BE
    # material lengths: 1m, 1m, 0.1m 0.1m, Be - ? 0.5? 0.2?
    abs_len = 0.2
    run_icool_sheet_mdl(sol_params=[0.350089], beam_params = beam, absorber = "LIH", abs_len=0.2)
    plot_energy_spread("for009.dat")
    dataframe, nfinal, emit_red = run_ecalc()
    # plot_bz("for009.dat")
    plot_from_ecalc(dataframe, nfinal, ninit)


# Emittance reduction
# Start emittance: 302.3, Emittance at the end: 263.6, Emittance reduction: 0.128018524644 %
# {'LHE': [0.8122394971882236, 0.8149189546807807, 0.21104862719153147, 0.1081706913661925, 0.07310618590803827, 0.05094277208071445, 0.04134965266291745], 'LH': [0.8240820377108832, 0.8796559708898445, 0.2848164075421766, 0.15216672179953672, 0.0922924247436321, 0.07476017201455479, 0.05755871650678128], 'LI': [0.43962950711214016, 0.7034072113794243, 0.8181276877274231, 0.8779027456169369, 0.3870327489249089, 0.19318557724115104, 0.12801852464439278]}

# Transmission: to be repeated!
# {'LHE': [0, 0, 1, 1, 1, 1, 1], 'LH': [0, 0, 0, 1, 1, 1, 1], 'LI': [0, 0, 0, 0, 0, 1, 1]}


#energy spread
# {'LHE': [nan, 2.590794969232273, 2.5424141392319672, 2.375566685963562, 2.2516533544924444, 2.0925462085209197, 1.9635139285480683], 'LH': [nan, nan, 2.5424141392319672, 2.361558754585717, 2.225701230860272, 2.094562073930133, 1.9770886910947945], 'LI': [nan, nan, nan, nan, 2.573097549435444, 2.743930187808943, 2.560288890793425]}


# optimal for p = 135Mev/c
# r_HF: 0.350089
# second result: [0.57057834]

# for p = 150MeV/c: 
# using optimization results for HF_Radius obtained with matching for 135 MeV/c (r_HF = 0.350089):
# Start emittance: 300.3, Emittance at the end: 229.6, Emittance reduction: 0.235431235431 %
# ---> worth to rerun the optimization? ---> TODO: try much lower energies

# after running optimization for this specific energy, to match the optics first --> r_HF = 0.5156793093558257:
# Start emittance: 300.3, Emittance at the end: 224.3, Emittance reduction: 0.25308025308 %

# second optimization run: r_HF = 0.31527386929048584
# max field: 36.4T, emittance reduction = 29%


#optimized for p=100MeV/c: radius = [0.30203018]
# 

# optimized for p = 0.075
# HF_radius: 0.5686278729515404, HF: 31.1T






