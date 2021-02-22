# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:06:06 2019

module: universal module - contains all the core input data - usually held constant/doesn't change between regions or farms'



@author: young
"""

##python modules
import pickle as pkl
import numpy as np

import Functions as fun


#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
import os.path
try:
    if os.path.getmtime("Universal.xlsx") > os.path.getmtime("pkl_universal.pkl"):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print('Reading universal inputs from pickle', end=' ', flush=True)
except FileNotFoundError:      
    inputs_from_pickle = False

filename= 'pkl_universal.pkl'
##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading universal inputs from Excel', end=' ', flush=True)
    with open(filename, "wb") as f:
        ##prices
        price_inp = fun.xl_all_named_ranges("Universal.xlsx","Price")
        pkl.dump(price_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##Finance inputs
        finance_inp = fun.xl_all_named_ranges("Universal.xlsx","Finance")
        pkl.dump(finance_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##mach inputs - general
        mach_general_inp = fun.xl_all_named_ranges("Universal.xlsx","Mach General")
        pkl.dump(mach_general_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##sup inputs
        sup_inp = fun.xl_all_named_ranges("Universal.xlsx","Sup Feed")
        pkl.dump(sup_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##crop inputs
        crop_inp = fun.xl_all_named_ranges("Universal.xlsx","Crop Sim")
        pkl.dump(crop_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##sheep inputs
        sheep_inp = fun.xl_all_named_ranges('Universal.xlsx', 'Sheep', numpy=True)
        pkl.dump(sheep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        parameters_inp = fun.xl_all_named_ranges('Universal.xlsx', 'Parameters', numpy=True)
        pkl.dump(parameters_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        pastparameters_inp = fun.xl_all_named_ranges('Universal.xlsx', 'PastParameters', numpy=True)
        pkl.dump(pastparameters_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##mach options
        ###create a dict to store all options - this allows the user to select an option
        machine_options_dict_inp={}
        machine_options_dict_inp[1] = fun.xl_all_named_ranges("Universal.xlsx","Mach 1")
        pkl.dump(machine_options_dict_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(filename, "rb") as f:
        price_inp = pkl.load(f)
        
        finance_inp = pkl.load(f)
        
        mach_general_inp = pkl.load(f)
        
        sup_inp = pkl.load(f)
        
        crop_inp = pkl.load(f)
        
        sheep_inp = pkl.load(f)
        
        parameters_inp = pkl.load(f)
        pastparameters_inp = pkl.load(f)
        
        machine_options_dict_inp  = pkl.load(f)
print('- finished')

##reshape require inputs
###lengths
len_h1 = sheep_inp['i_husb_muster_infrastructurereq_h1h4'].shape[-1]
len_h4 = sheep_inp['i_h4_len']
len_h6 = sheep_inp['i_husb_muster_requisites_prob_h6h4'].shape[-1]
len_l2 = sheep_inp['i_husb_muster_labourreq_l2h4'].shape[-1]
len_m4 = sheep_inp['i_salep_months_priceadj_s7s9m4'].shape[-1]
len_s5 = sheep_inp['i_s5_len'],
len_s6 = sheep_inp['i_salep_score_scalar_s7s5s6'].shape[-1]
len_s7 = sheep_inp['i_s7_len']
len_s9 = sheep_inp['i_s9_len']




###shapes
h4h1 = (len_h4, len_h1)
h4h6 = (len_h4, len_h6)
h4l2 = (len_h4, len_l2)
s7s9m4 = (len_s7, len_s9, len_m4)
s7s5s6 = (len_s7, len_s5, len_s6)
s7s5 = (len_s7, len_s5)
cb0 = (parameters_inp['i_cb0_len'], parameters_inp['i_cb0_len2'],-1)
ce = (parameters_inp['i_ce_len'], parameters_inp['i_ce_len2'],-1)
cl0 = (parameters_inp['i_cl0_len'], parameters_inp['i_cl0_len2'],-1)
cl1 = (parameters_inp['i_cl1_len'], parameters_inp['i_cl1_len2'],-1)
cu1 = (parameters_inp['i_cu1_len'], parameters_inp['i_cu1_len2'],-1)
cu2 = (parameters_inp['i_cu2_len'], parameters_inp['i_cu2_len2'],-1)
cx = (parameters_inp['i_cx_len'], parameters_inp['i_cx_len2'],-1)

###stock
sheep_inp['i_salep_months_priceadj_s7s9m4'] = np.reshape(sheep_inp['i_salep_months_priceadj_s7s9m4'], s7s9m4)
sheep_inp['i_salep_score_scalar_s7s5s6'] = np.reshape(sheep_inp['i_salep_score_scalar_s7s5s6'], s7s5s6)
sheep_inp['i_salep_weight_scalar_s7s5s6'] = np.reshape(sheep_inp['i_salep_weight_scalar_s7s5s6'], s7s5s6)
sheep_inp['i_salep_weight_range_s7s5'] = np.reshape(sheep_inp['i_salep_weight_range_s7s5'], s7s5)
sheep_inp['i_husb_muster_requisites_prob_h6h4'] = np.reshape(sheep_inp['i_husb_muster_requisites_prob_h6h4'], h4h6)
sheep_inp['i_husb_muster_labourreq_l2h4'] = np.reshape(sheep_inp['i_husb_muster_labourreq_l2h4'], h4l2)
sheep_inp['i_husb_muster_infrastructurereq_h1h4'] = np.reshape(sheep_inp['i_husb_muster_infrastructurereq_h1h4'], h4h1)
parameters_inp['i_cb0_c2'] = np.reshape(parameters_inp['i_cb0_c2'], cb0)
parameters_inp['i_cb0_y'] = np.reshape(parameters_inp['i_cb0_y'], cb0)
parameters_inp['i_ce_c2'] = np.reshape(parameters_inp['i_ce_c2'], ce)
parameters_inp['i_ce_y'] = np.reshape(parameters_inp['i_ce_y'], ce)
parameters_inp['i_cl0_c2'] = np.reshape(parameters_inp['i_cl0_c2'], cl0)
parameters_inp['i_cl0_y'] = np.reshape(parameters_inp['i_cl0_y'], cl0)
parameters_inp['i_cl1_c2'] = np.reshape(parameters_inp['i_cl1_c2'], cl1)
parameters_inp['i_cl1_y'] = np.reshape(parameters_inp['i_cl1_y'], cl1)
parameters_inp['i_cu1_c2'] = np.reshape(parameters_inp['i_cu1_c2'], cu1)
parameters_inp['i_cu1_y'] = np.reshape(parameters_inp['i_cu1_y'], cu1)
parameters_inp['i_cu2_c2'] = np.reshape(parameters_inp['i_cu2_c2'], cu2)
parameters_inp['i_cu2_y'] = np.reshape(parameters_inp['i_cu2_y'], cu2)
parameters_inp['i_cx_c2'] = np.reshape(parameters_inp['i_cx_c2'], cx)
parameters_inp['i_cx_y'] = np.reshape(parameters_inp['i_cx_y'], cx)

##pasture
pastparameters_inp['i_cu3_c4'] = pastparameters_inp['i_cu3_c4'].reshape(pastparameters_inp['i_cu3_len'], pastparameters_inp['i_cu3_len2'], -1)
pastparameters_inp['i_cu4_c4'] = pastparameters_inp['i_cu4_c4'].reshape(pastparameters_inp['i_cu4_len'], pastparameters_inp['i_cu4_len2'], -1)


##copy inputs so there is an origional (before SA) version
price = price_inp.copy()
finance = finance_inp.copy()
mach_general = mach_general_inp.copy()
supfeed = sup_inp.copy()
crop = crop_inp.copy()
sheep = sheep_inp.copy()
parameters = parameters_inp.copy()
pastparameters = pastparameters_inp.copy()
mach = machine_options_dict_inp.copy()

#######################
#apply SA             #
#######################
def universal_inp_sa():
    '''
    
    Returns
    -------
    None.
    
    Applies sensitivity adjustment to each input.
    This function gets called at the beginning of each loop in the exp.py module

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen 
    ##enter sa below

    ##sheep
    ###SAT
    sheep['i_salep_weight_scalar_s7s5s6'] = fun.f_sa(sheep_inp['i_salep_weight_scalar_s7s5s6'], sen.sat['salep_weight_scalar'], 3, 1, 0) #Scalar for LW impact across grid 1 (sat adjusted)
    sheep['i_salep_score_scalar_s7s5s6'] = fun.f_sa(sheep_inp['i_salep_score_scalar_s7s5s6'], sen.sat['salep_score_scalar'], 3, 1, 0) #Scalar for score impact across the grid (sat adjusted)
    ###SAV
    sheep['i_eqn_compare'] = fun.f_sa(sheep_inp['i_eqn_compare'], sen.sav['eqn_compare'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g0_q1p7'] = fun.f_sa(sheep_inp['i_eqn_used_g0_q1p7'], sen.sav['eqn_used_g0_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g1_q1p7'] = fun.f_sa(sheep_inp['i_eqn_used_g1_q1p7'], sen.sav['eqn_used_g1_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g2_q1p7'] = fun.f_sa(sheep_inp['i_eqn_used_g2_q1p7'], sen.sav['eqn_used_g2_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g3_q1p7'] = fun.f_sa(sheep_inp['i_eqn_used_g3_q1p7'], sen.sav['eqn_used_g3_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_woolp_mpg_percentile'] = fun.f_sa(sheep_inp['i_woolp_mpg_percentile'], sen.sav['woolp_mpg_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_woolp_fdprem_percentile'] = fun.f_sa(sheep_inp['i_woolp_fdprem_percentile'], sen.sav['woolp_fdprem_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_salep_percentile'] = fun.f_sa(sheep_inp['i_salep_percentile'], sen.sav['salep_percentile'], 5) #Value for percentile for all sale grids



##############
#need to be added to spreadsheet - added by mry when adding dvps
##############

# structure['i_fvp4_date_i'] = np.array([np.datetime64('2019-11-15'), np.datetime64('2019-01-15')])
#                                         #prejoin, others..............
# # structure['i_fvp_mask_dams'] = np.array([True, True, True, True,	False]) #prejoining dvp must always be True. (dvp from start of sim to first other dvp is not included - it is added as a true in code)
# structure['i_dvp_mask_f1'] = np.array([True, True, True, False,	False]) #prejoining dvp must always be True. (dvp from start of sim to first other dvp is not included - it is added as a true in code)
# structure['i_w_start_len1'] = 3
#
# structure['ia_r1type_fi'] = np.array([[0,0], #dont need to add ones that are not dvp (eg just leave them as 2 (default))
#                                      [1,1],
#                                      [2,2],
#                                      [2,2],
#                                      [2,2]]) #todo add this input also make sure user changes this when changing dvp timing.
'''The number is the number of fvps prior to the start of this dvp. So if the dvp dates are say
1 Feb, 1 May & 1 July and the FVP dates are
1 Feb, 1 May, 1 June, 1 July, 1 Oct. Then 
n_prior_damfvps_v = 0, 1, 3
n_damsfvps_v = 1,2,2'''
# #number of fvps that occur during a dvp
# structure['i_n_fvps_vi1'] = np.array([[1,1], #this is only the v type axis. it is expanded to full v axis in code.
#                                      [1,1],
#                                      [2,2]])
# #number of fvps since condensing. this would need to change if fvp/dvp aded or removed or changes date.
# structure['i_n_prior_fvps_vi1'] = np.array([[0,0], #this is only the v type axis. it is expanded to full v axis in code.
#                                              [1,1],
#                                              [2,2]])

##need to alter how these are handle
#todo remove the inputs below.
#structure['i_n_fvp_period1']
# structure['i_w1_len']
# structure['i_w_idx_dams']
#todo these need to be calibrated - these are mask to be the correct length but need to ensure that the values are correct for the number of dvps
structure['i_adjp_lw_initial_w1'] = np.array([0.0, 0.15, -0.15])
structure['i_adjp_cfw_initial_w1'] = np.array([0.0, 0.10, -0.10])
structure['i_adjp_fd_initial_w1'] = np.array([0.0, 0.5, -0.5])
structure['i_adjp_fl_initial_w1'] = np.array([0.0, 0.10, -0.10])




