"""

These are inputs that are expected to remain constant between regions and properties, this includes:

* Prices of inputs
* Value of outputs (grain, wool ,meat)
* Interest rates & depreciation rates
* Machinery options
* Sheep parameters and definition of genotypes

author: young
"""

##python modules
import pickle as pkl
import numpy as np
import copy

import Functions as fun


#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
import os.path

##build path this way so that readthedocs can read correctly.
directory_path = os.path.dirname(os.path.abspath(__file__))
universal_xl_path = os.path.join(directory_path, "Universal.xlsx")
universal_pkl_path = os.path.join(directory_path, "pkl_universal.pkl")

try:
    if os.path.getmtime(universal_xl_path) > os.path.getmtime(universal_pkl_path):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print('Reading universal inputs from pickle', end=' ', flush=True)
except FileNotFoundError:      
    inputs_from_pickle = False

##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading universal inputs from Excel', end=' ', flush=True)
    with open(universal_pkl_path, "wb") as f:
        ##prices
        price_inp = fun.xl_all_named_ranges(universal_xl_path,"Price")
        pkl.dump(price_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##Finance inputs
        finance_inp = fun.xl_all_named_ranges(universal_xl_path,"Finance")
        pkl.dump(finance_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##mach inputs - general
        mach_general_inp = fun.xl_all_named_ranges(universal_xl_path,"Mach General")
        pkl.dump(mach_general_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        ##sup inputs
        sup_inp = fun.xl_all_named_ranges(universal_xl_path,"Sup Feed")
        pkl.dump(sup_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##crop inputs
        crop_inp = fun.xl_all_named_ranges(universal_xl_path,"Crop Sim")
        pkl.dump(crop_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##sheep inputs
        sheep_inp = fun.xl_all_named_ranges(universal_xl_path, 'Sheep', numpy=True)
        pkl.dump(sheep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        parameters_inp = fun.xl_all_named_ranges(universal_xl_path, 'Parameters', numpy=True)
        pkl.dump(parameters_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        pastparameters_inp = fun.xl_all_named_ranges(universal_xl_path, 'PastParameters', numpy=True)
        pkl.dump(pastparameters_inp, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        ##mach options
        ###create a dict to store all options - this allows the user to select an option
        machine_options_dict_inp={}
        machine_options_dict_inp[1] = fun.xl_all_named_ranges(universal_xl_path,"Mach 1")
        pkl.dump(machine_options_dict_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(universal_pkl_path, "rb") as f:
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
len_p4 = sheep_inp['i_salep_months_priceadj_s7s9p4'].shape[-1]
len_s5 = sheep_inp['i_s5_len']
len_s6 = sheep_inp['i_salep_score_scalar_s7s5s6'].shape[-1]
len_s7 = sheep_inp['i_s7_len']
len_s9 = sheep_inp['i_s9_len']




###shapes
h4h1 = (len_h4, len_h1)
h4h6 = (len_h4, len_h6)
h4l2 = (len_h4, len_l2)
s7s9p4 = (len_s7, len_s9, len_p4)
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
sheep_inp['i_salep_months_priceadj_s7s9p4'] = np.reshape(sheep_inp['i_salep_months_priceadj_s7s9p4'], s7s9p4)
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


##create a copy of each input dict - so that the base inputs remain unchanged
##the copy created is the one used in the actual modules
###NOTE: if an input sheet is added remember to add it to the dict reset in f_sa() below.
price = copy.deepcopy(price_inp)
finance = copy.deepcopy(finance_inp)
mach_general = copy.deepcopy(mach_general_inp)
supfeed = copy.deepcopy(sup_inp)
crop = copy.deepcopy(crop_inp)
sheep = copy.deepcopy(sheep_inp)
parameters = copy.deepcopy(parameters_inp)
pastparameters = copy.deepcopy(pastparameters_inp)
mach = copy.deepcopy(machine_options_dict_inp)

#######################
#apply SA             #
#######################
def universal_inp_sa():
    '''
    Applies sensitivity adjustment to each input.
    This function gets called at the beginning of each loop in the exp.py module.

    SA order is: sav, sam, sap, saa, sat, sar.
    So that multiple SA can be applied to one input.

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen 

    ##reset inputs to base at the start of each trial before applying SA  - old method was to update the SA based on the _inp dict but that doesnt work well when multiple SA on the same variable.
    fun.f_dict_reset(price, price_inp)
    fun.f_dict_reset(finance, finance_inp)
    fun.f_dict_reset(mach_general, mach_general_inp)
    fun.f_dict_reset(supfeed, sup_inp)
    fun.f_dict_reset(crop, crop_inp)
    fun.f_dict_reset(sheep, sheep_inp)
    fun.f_dict_reset(parameters, parameters_inp)
    fun.f_dict_reset(pastparameters, pastparameters_inp)
    fun.f_dict_reset(mach, machine_options_dict_inp)

    ##finance
    ###SAV
    finance['minroe'] = fun.f_sa(finance['minroe'], sen.sav['minroe'], 5)  #value for minroe (same sav as below)
    finance['minroe_dsp'] = fun.f_sa(finance['minroe_dsp'], sen.sav['minroe'], 5)  #value for minroe (same sav as above)

    ##price
    ###sav
    price['grain_price_percentile'] = fun.f_sa(price['grain_price_percentile'],sen.sav['grain_percentile'], 5)


    ##sheep
    ###SAV
    sheep['i_eqn_compare'] = fun.f_sa(sheep['i_eqn_compare'], sen.sav['eqn_compare'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g0_q1p7'] = fun.f_sa(sheep['i_eqn_used_g0_q1p7'], sen.sav['eqn_used_g0_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g1_q1p7'] = fun.f_sa(sheep['i_eqn_used_g1_q1p7'], sen.sav['eqn_used_g1_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g2_q1p7'] = fun.f_sa(sheep['i_eqn_used_g2_q1p7'], sen.sav['eqn_used_g2_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g3_q1p7'] = fun.f_sa(sheep['i_eqn_used_g3_q1p7'], sen.sav['eqn_used_g3_q1p7'], 5)  #determines if both equation systems are being run and compared
    sheep['i_woolp_mpg_percentile'] = fun.f_sa(sheep['i_woolp_mpg_percentile'], sen.sav['woolp_mpg_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_woolp_fdprem_percentile'] = fun.f_sa(sheep['i_woolp_fdprem_percentile'], sen.sav['woolp_fdprem_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_salep_percentile'] = fun.f_sa(sheep['i_salep_percentile'], sen.sav['salep_percentile'], 5) #Value for percentile for all sale grids
    ###SAM
    sheep['i_husb_operations_contract_cost_h2'] = fun.f_sa(sheep['i_husb_operations_contract_cost_h2'],sen.sam['husb_cost_h2'])
    sheep['i_husb_operations_muster_propn_h2'] = fun.f_sa(sheep['i_husb_operations_muster_propn_h2'], sen.sam['husb_mustering_h2'])
    sheep['i_husb_operations_labourreq_l2h2'] = fun.f_sa(sheep['i_husb_operations_labourreq_l2h2'],sen.sam['husb_labour_l2h2'])
    sheep['i_salep_months_priceadj_s7s9p4'] = fun.f_sa(sheep['i_salep_months_priceadj_s7s9p4'],sen.sam['salep_month_adjust_s7s9p4'])
    ###SAP
    ###SAA
    sheep['i_husb_operations_contract_cost_h2'] = fun.f_sa(sheep['i_husb_operations_contract_cost_h2'],sen.saa['husb_cost_h2'], 2)
    sheep['i_husb_operations_labourreq_l2h2'] = fun.f_sa(sheep['i_husb_operations_labourreq_l2h2'],sen.saa['husb_labour_l2h2'], 2)
    ###SAT
    sheep['i_salep_weight_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_weight_scalar_s7s5s6'], sen.sat['salep_weight_scalar'], 3, 1, 0) #Scalar for LW impact across grid 1 (sat adjusted)
    sheep['i_salep_score_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_score_scalar_s7s5s6'], sen.sat['salep_score_scalar'], 3, 1, 0) #Scalar for score impact across the grid (sat adjusted)
    ###SAR

    ##parameters (c2 genotype sensitivity)
    ###SAV - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_srw_c2'] = fun.f_sa(parameters['i_srw_c2'].astype(float), sen.sav['srw_c2'], 5) #genotype srw
    parameters['i_ce_c2'][2,...] = fun.f_sa(parameters['i_ce_c2'][2,...].astype(float), sen.sav['bnd_twice_dry_propn'], 5) #propn of twice drys
    parameters['i_cl0_c2'] = fun.f_sa(parameters['i_cl0_c2'].astype(float), sen.sav['cl0_c2'], 5) #genotype litter size params
    ###SAM - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_ci_c2'] = fun.f_sa(parameters['i_ci_c2'].astype(float),sen.sam['ci_c2'])
    parameters['i_sfw_c2'] = fun.f_sa(parameters['i_sfw_c2'].astype(float),sen.sam['sfw_c2'])
    ###SAP
    ###SAA - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_sfd_c2'] = fun.f_sa(parameters['i_sfd_c2'].astype(float),sen.saa['sfd_c2'], 2)
    parameters['i_cl0_c2'] = fun.f_sa(parameters['i_cl0_c2'].astype(float), sen.saa['cl0_c2'], 2) #genotype litter size params
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float), sen.saa['scan_std_c2'], 2) #genotype scanning percent params

    ##parameters (overall sensitivity - carried out after the c2 genotype sa)
    ###SAM - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float), sen.sam['rr']) #genotype scanning percent params
    ###SAA - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float), sen.saa['rr'], 2
                                           ) * (parameters['i_scan_std_c2'] > 0)  #stays as zero if original value was zero


