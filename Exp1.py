# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: experiment module - this is the module that runs everything and controls kv's

@author: young
"""


import pandas as pd
import pyomo.environ as pe
import time
import math
import os
import os.path
import sys
from datetime import datetime
import multiprocessing
import pickle as pkl
import json
import numpy as np

#report the clock time that the experiment was started
print(f'Experiment commenced at: {time.ctime()}')
start = time.time()

import CreateModel as crtmod
import BoundsPyomo as bndpy
import StructuralInputs as sinp
import UniversalInputs as uinp
import PropertyInputs as pinp
import Sensitivity as sen
import Functions as fun
import RotationPyomo as rotpy
import Crop as crp
import CropPyomo as crppy
import MachPyomo as macpy
import FinancePyomo as finpy
import LabourFixedPyomo as lfixpy
import LabourPyomo as labpy
import LabourCropPyomo as lcrppy
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
import StockPyomo as spy
import CorePyomo as core
import MVF as mvf

##used to get status on multiprocessing
# import logging
# logger = multiprocessing.log_to_stderr(logging.DEBUG)

## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
try:
    maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
except IndexError:  # in case no arg passed to python
    maximum_processes = 1  # available memory / value determined by size of the model being run (~5GB for the small model)


    
#########################
#load exp               # 
#########################
##read in exp and drop all false runs ie runs not being run this time
exp_data, exp_group_bool = fun.f_read_exp()
exp_data1=exp_data.copy() #copy made so that the run col can be added - the original df is used to allocate sa values (would cause an error if run col existed but i can't drop it because it is used to determine if the trial is run)


##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xls has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial
exp_data1 = fun.f_run_required(exp_data1)

if __name__ == '__main__':
    ##check Output and pkl folders exist for outputs. If not create.
    if os.path.isdir('pkl'):
        pass
    else:
        os.mkdir('pkl')
    if os.path.isdir('Output'):
        pass
    else:
        os.mkdir('Output')
    if os.path.isdir('Output/infeasible'):
        pass
    else:
        os.mkdir('Output/infeasible')

    ##plk a copy of exp. Used next time the model is run to identify which trials are up to date.
    with open('pkl/pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)

##cut exp_data based on the experiment group
exp_data = fun.f_group_exp(exp_data, exp_group_bool)
exp_data1 = fun.f_group_exp(exp_data1, exp_group_bool)

## Define the dataset - trials that require at least the precalcs done (user wants it run and it is out of date)
dataset = list(np.flatnonzero(np.nan_to_num(np.array(exp_data.index.get_level_values(0))) * np.array(exp_data1['run_req'])))  # gets the ordinal index values for the trials the user wants to run that are not up to date
## number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
n_processes = min(multiprocessing.cpu_count(),len(dataset),maximum_processes)

## set the start time at the beginning of the multiprocessing loops
start_time1 = time.time()


#########################
#Exp loop               #
#########################

def exp(row):  # called with command: pool.map(exp, dataset)

    ##can use logger to get status on multiprocessing
    # logger.info('Received {}'.format(row))

    ##start timer for each loop
    start_time = time.time()

    ##check the rotations and inputs align - this means rotation method can be controlled using a SA
    crp.f_rot_check()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    trial_description = f'{dataset.index(row)+1} {trial_name}'
    print(f'\n{trial_description}, Starting trial at: {time.ctime()}')

    ##update sensitivity values
    fun.f_update_sen(row,exp_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)

    ##call sa functions - assigns sa variables to relevant inputs
    sinp.structural_inp_sa()
    uinp.universal_inp_sa()
    pinp.property_inp_sa()
    ##create empty dicts - have to do it here because need the trial as the first key, so whole trial can be compared when determining if pyomo needs to be run
    ###params
    params={}
    params['pas']={}
    params['rot']={}
    params['crop']={}
    params['mach']={}
    params['fin']={}
    params['labfx']={}
    params['lab']={}
    params['crplab']={}
    params['sup']={}
    params['stub']={}
    params['stock']={}
    ###report values
    r_vals={}
    r_vals['pas']={}
    r_vals['rot']={}
    r_vals['crop']={}
    r_vals['mach']={}
    r_vals['fin']={}
    r_vals['labfx']={}
    r_vals['lab']={}
    r_vals['crplab']={}
    r_vals['sup']={}
    r_vals['stub']={}
    r_vals['stock']={}
    nv = {} #dict to store nv params from StockGenerator to be used in pasture

    ##call precalcs
    precalc_start = time.time()
    rotpy.rotation_precalcs(params['rot'],r_vals['rot'])
    crppy.crop_precalcs(params['crop'],r_vals['crop'])
    macpy.mach_precalcs(params['mach'],r_vals['mach'])
    finpy.fin_precalcs(params['fin'],r_vals['fin'])
    lfixpy.labfx_precalcs(params['labfx'],r_vals['labfx'])
    labpy.lab_precalcs(params['lab'],r_vals['lab'])
    lcrppy.crplab_precalcs(params['crplab'],r_vals['crplab'])
    suppy.sup_precalcs(params['sup'],r_vals['sup'])
    spy.stock_precalcs(params['stock'],r_vals['stock'],nv)
    stubpy.stub_precalcs(params['stub'],r_vals['stub'], nv) #stub must be after stock because it uses nv dict which is populated in stock.py
    paspy.paspyomo_precalcs(params['pas'],r_vals['pas'], nv) #pas must be after stock because it uses nv dict which is populated in stock.py
    precalc_end = time.time()
    print(f'{trial_description}, total time for precalcs: {precalc_end - precalc_start:.2f} finished at {time.ctime()}')

    ##does pyomo need to be run? In exp1 pyomo is always run because creating params file take up lots of time, RAM and disc space
    run_pyomo_params = True

    ##determine if pyomo should run, note if pyomo doesn't run there will be no full solution (they are the same as before so no need)
    if run_pyomo_params:
        ##call core model function, must call them in the correct order (core must be last)
        pyomocalc_start = time.time()
        model = pe.ConcreteModel() #create pyomo model - done each loop because memory was being leaked when just deleting and re adding the components.
        crtmod.sets(model, nv) #certain sets have to be updated each iteration of exp
        rotpy.rotationpyomo(params['rot'], model)
        crppy.croppyomo_local(params['crop'], model)
        macpy.machpyomo_local(params['mach'], model)
        finpy.finpyomo_local(params['fin'], model)
        lfixpy.labfxpyomo_local(params['labfx'], model)
        labpy.labpyomo_local(params['lab'], model)
        lcrppy.labcrppyomo_local(params['crplab'], model)
        paspy.paspyomo_local(params['pas'], model)
        suppy.suppyomo_local(params['sup'], model)
        stubpy.stubpyomo_local(params['stub'], model)
        spy.stockpyomo_local(params['stock'], model)
        mvf.mvf_pyomo(model)
        ###bounds-this must be done last because it uses sets built in some of the other modules
        bndpy.boundarypyomo_local(params, model)
        pyomocalc_end = time.time()
        print(f'{trial_description}, time for localpyomo: {pyomocalc_end - pyomocalc_start:.2f} finished at {time.ctime()}')
        obj = core.coremodel_all(params, trial_name, model)
        print(f'{trial_description}, time for corepyomo: {time.time() - pyomocalc_end:.2f} finished at {time.ctime()}')

        if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
            ##This writes variable summary each iteration with generic file name - it is overwritten each iteration and is created so the run progress can be monitored
            fun.write_variablesummary(model, row, exp_data, obj, 1)

            ##check if user wants full solution
            if exp_data.index[row][1] == True:
                ##make lp file
                model.write('Output/%s.lp' %trial_name,io_options={'symbolic_solver_labels':True})  #file name has to have capital

                ##This writes variable summary for full solution (same file as the temporary version created above)
                fun.write_variablesummary(model, row, exp_data, obj)

                ##prints what you see from pprint to txt file - you can see the slack on constraints but not the rc or dual
                with open('Output/Full model - %s.txt' %trial_name, 'w') as f:  #file name has to have capital
                    f.write("My description of the instance!\n")
                    model.display(ostream=f)

                ##write rc, duals and slacks to txt file. Duals are slow to write so that option must be turn on
                write_duals = True
                with open('Output/Rc and Duals - %s.txt' %trial_name,'w') as f:  #file name has to have capital
                    f.write('RC\n')
                    for v in model.component_objects(pe.Var, active=True):
                        f.write("Variable %s\n" %v)   
                        for index in v:
                            try: #in case variable has no index
                                print("      ", index, model.rc[v[index]], file=f)
                            except: pass
                    f.write('Slacks (no entry means no slack)\n')  # this can be used in search to find the start of this in the txt file
                    for c in model.component_objects(pe.Constraint,active=True):
                        f.write("Constraint %s\n" % c)
                        for index in c:
                            if c[index].lslack() != 0 and c[index].lslack() != np.inf:
                                print("  L   ",index,c[index].lslack(),file=f)
                            if c[index].uslack() != 0 and c[index].lslack() != np.inf:
                                print("  U   ",index,c[index].uslack(),file=f)
                    if write_duals:
                        f.write('Dual\n')   #this can be used in search to find the start of this in the txt file
                        for c in model.component_objects(pe.Constraint, active=True):
                            f.write("Constraint %s\n" %c)
                            for index in c:
                                print("      ", index, model.dual[c[index]], file=f)

            season = pinp.f_keys_z()[0]
            lp_vars = {}
            variables=model.component_objects(pe.Var, active=True)
            lp_vars[season] = {str(v):{s:v[s].value for s in v} for v in variables}     #creates dict with variable in it. This is tricky since pyomo returns a generator object
            lp_vars[season]['scenario_profit'] = obj
            ##store profit
            lp_vars['profit'] = obj

        ## if DSP version
        else:
            ##Note: to get full lp file for DSP model it needs to be run via the terminal (or in runef.py) see google doc for more info.

            ## load json (dsp solution) back in
            with open('ef_solution.json') as f:
                data = json.load(f)

            ##store pyomo variable output as a dict
            lp_vars = {}
            for season in pinp.f_keys_z():
                variables = data['scenario solutions'][season]['variables']
                lp_vars[season] = {}
                ###get dict into correct format for lp_vars
                for key in variables.keys():
                    var_name = key.split('[', 1)[0]
                    try:
                        sets = key.split('[', 1)[1].split(']',1)[0]
                        sets = tuple(sets.split(','))
                    except IndexError: #handle variables with no sets
                        sets = None #set to None because this happens in steady state version.
                    value = variables[key]['value']
                    try:
                        lp_vars[season][var_name][sets] = value
                    except KeyError:
                        lp_vars[season][var_name] = {} #create empty dict once for each variable name
                        lp_vars[season][var_name][sets] = value

                ##store scenario profit
                lp_vars[season]['scenario_profit'] = data['scenario solutions'][season]['objective']
            ##store overall expected profit
            lp_vars['profit'] = obj

        ##pickle lp info - only if pyomo is run
        with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"wb") as f:
            pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)
    ##pickle report values - every time a trial is run
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"wb") as f:
        pkl.dump(r_vals,f,protocol=pkl.HIGHEST_PROTOCOL)

    #last step is to print the time for the current trial to run
    ##determine expected time to completion - trials left multiplied by average time per trial
    ##this approach works well if chunksize = 1
    total_batches = math.ceil(len(dataset) / n_processes)
    current_batch = math.ceil((dataset.index(row)+1) / n_processes) #add 1 because python starts at 0
    remaining_batches = total_batches - current_batch
    time_taken = time.time() - start_time1   #start of the multiprocessing loops
    batch_time = time_taken / current_batch
    time_remaining = remaining_batches * batch_time
    finish_time_expected = time.time() + time_remaining

    # ## determine expected time to completion - total time is time this loop multiplied by the number of batches
    loop_time = time.time() - start_time
    # ## finish time if all batches take the same time as this loop.
    # ## This approach underestimates the final time if this loop was quicker than average
    # ## not accurate if the experiment has trials with different model specifications (scan, F or N)
    # finish_time_expected = start_time1 + loop_time * total_batches

    print(f'{trial_description}, total time taken this loop: {loop_time:.2f}')
    message = f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m at {time.ctime()}'
    #replace message if this process is complete
    if remaining_batches == 0:
        message = f'{trial_description}, this process is complete'
    print(message)

    return row

##works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#   using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each iteration returns a dict and the multiprocess stuff returns a list)
def main():
    ##displays start status - number of trials to run, date and time exp.xl was last saved and output summary
    print(f'Number of trials to run: {len(dataset)}')
    print(f'Number of full solutions: {sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data)))}')
    print(f'Exp.xls last saved: {datetime.fromtimestamp(round(os.path.getmtime("exp.xlsx")))}')
    ##start multiprocessing
    with multiprocessing.Pool(processes=n_processes) as pool:
        ##size 1 has similar speed even for N11 model and allows better reporting (will be even better on a larger model)
        ##a drawback of chunksize = 1 is that if there is an error in the multiprocessed code then every trial is still processed
        trials_successfully_run = pool.map(exp, dataset, chunksize = 1)

    return

if __name__ == '__main__':
    main() #returns a list of dicts in the order of exp
    end = time.time()
    print(f'\n\033[1mExperiment completed at:\033[0m {time.ctime()}, total time taken: {end - start:.2f}')
    try:
        print(f'average time taken for each loop: {(end - start) / len(dataset):.2f}')  #average time since start of experiment
    except ZeroDivisionError:
        pass
