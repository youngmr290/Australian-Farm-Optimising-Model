# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young

This is where all sensitivity values must be initialised.

"""

import numpy as np
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Periods as per

##create dict - store sa variables in dict so they can easily be changed in the exp loop
sam = dict()
sap = dict()
saa = dict()
sat = dict()
sav = dict()
sar = dict()

######
#SAP #
######

##Global
sap['pi']=0 #global potential intake (this increases animal intake without altering animal energy profile, to alter the energy profile use ci[1:2,...] todo this has not been applied globally to the model yet (currently only in stubble)

######
#SAM #
######
##general
sam['random'] = 1.0   # SA multiplier used to tweak any random variable when debugging or checking something (after being used it is best to remove it)


## Annual module sensitivity variables - these need to have the same name for each pasture type
sam['germ','annual']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
sam['germ_l','annual']                  = np.ones((len(pinp.general['lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam['pgr','annual']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
sam['pgr_f','annual']                   = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam['pgr_l','annual']                   = np.ones((len(pinp.general['lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam['dry_dmd_decline','annual']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
sam['grn_dmd_declinefoo_f','annual']    = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam['grn_dmd_range_f','annual']         = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam['grn_dmd_senesce_f','annual']       = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
# sa_feed_period_inc_t      = True    # growth of this pasture in this period is included
# sa_lmu_inc_t              = True    # this pasture is included on this lmu

##livestock
sam['woolp_mpg'] = 1.0                      # sa multiplier for wool price at std micron
sam['salep_max'] = 1.0                      #max sale price in grid
sam['salep_month_adjust_s7s9p4'] = np.ones(uinp.sheep['i_salep_months_priceadj_s7s9p4'].shape, dtype=np.float64)      #monthly sale price
sam['kg'] = 1.0                             #energy efficiency of adults (zf2==1)
sam['mr'] = 1.0                             #Maintenance requirement of adults (zf2==1)
sam['pi'] = 1.0                             #Potential intake of adults (zf2==1)
sam['LTW_dams'] = 1.0                       #adjust impact of life time wool fleece effects
sam['LTW_offs'] = 1.0                       #adjust impact of life time wool fleece effects
sam['pi_post'] = 1.0                        #Post loop potential intake of adults (zf2==1)
sam['chill'] = 1.0                        #intermediate sam on chill.

##stock parameters
sam['ci_c2'] = np.ones(uinp.parameters['i_ci_c2'].shape, dtype=np.float64)  #intake params for genotypes
sam['sfw_c2'] = 1.0                         #std fleece weight genotype params
sam['rr'] = 1.0                        #scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
sam['husb_cost_h2'] = np.ones(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
sam['husb_mustering_h2'] = np.ones(uinp.sheep['i_husb_operations_muster_propn_h2'].shape, dtype=np.float64)  #SA value for mustering required for husbandry operations.
sam['husb_labour_l2h2'] = np.ones(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour

######
#SAP #
######
sap['evg'] = 0.0               #energy content of liveweight gain - this is a high level sa, it impacts within a calculation not on an input and is only implemented on adults
sap['mortalityp'] = 0.0        #Scale the calculated progeny mortality at birth relative - this is a high level sa, it impacts within a calculation not on an input
sap['mortalitye'] = 0.0        #Scale the calculated dam mortality at birth - this is a high level sa, it impacts within a calculation not on an input
sap['mortalityb'] = 0.0        #Scale the calculated base mortality - this is a high level sa, it impacts within a calculation not on an input
sap['kg_post'] = 0.0           #Post loop energy efficiency of adults (zf2==1)
sap['mr_post'] = 0.0           #Post loop maintenance requirement of adults (zf2==1)


######
#SAA #
######
##stock
saa['husb_cost_h2'] = np.zeros(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
saa['husb_labour_l2h2'] = np.zeros(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour
saa['r1_izg1'] = np.zeros(pinp.sheep['ia_r1_zig1'].shape, dtype=int)   #SA to change the base feed option selected for dams
saa['r1_izg3'] = np.zeros(pinp.sheep['ia_r1_zig3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
saa['feedsupply_r1jp'] = np.zeros(pinp.feedsupply['i_feedsupply_options_r1pj0'].shape, dtype=np.float64)  #SA value for feedsupply.
saa['feedsupply_adj_r2p'] = np.zeros(pinp.feedsupply['i_feedsupply_adj_options_r2p'].shape, dtype=np.float64)  #SA value for feedsupply adjustment.

##stock parameters
saa['sfd_c2'] = 0.0                     #std fibre diameter genotype params
saa['cl0_c2'] = np.zeros(uinp.parameters['i_cl0_c2'].shape, dtype=np.float64)  #SA value for litter size genotype params.
saa['scan_std_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
saa['rr'] = 0.0                    #reproductive rate/scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
saa['rr_age_og1'] = np.zeros(pinp.sheep['i_scan_og1'].shape, dtype=np.float64)    # reproductive rate by age. Use shape that has og1
saa['mortalityx'] = np.zeros(np.max(sinp.stock['a_nfoet_b1'])+1, dtype=np.float64)  #Adjust the progeny mortality due to exposure at birth relative - this is a high level sa, it impacts within a calculation not on an input

######
#SAT #
######
sat['salep_weight_scalar'] = 0.0 #Scalar for LW impact across grid 1
sat['salep_score_scalar'] = 0.0  #Scalar for score impact across the grid


######
#SAR #
######


######
#SAV #
######
##if you initialise an array it must be type object (so that you can assign int/float/bool into the array)
##general
sav['steady_state']      = '-'                  #SA to alter if the model is steady state
sav['rev_create']      = '-'                  #SA to alter if the trial is being used to create rev std values
sav['rev_number']      = '-'                  #SA to alter rev number - rev number is appended to the std rev value pkl file and can be used to select which rev is used as std for a given trial.
sav['rev_trait_inc'] = np.full_like(sinp.structuralsa['rev_trait_inc'], '-', dtype=object) #SA value for which traits are to be held constant in REV analysis.

##finance
sav['minroe']      = '-'                  #SA to alter the minroe (applied to both steady-state and dsp minroe inputs)

##price
sav['grain_percentile'] = '-'  #grain price percentile


##bounds
sav['bnd_total_pas_area'] = '-'  #Total pasture area for bound. '-' is default so it will chuck an error if the bound is turned on without a specified area
sav['bnd_pasarea_inc'] = '-'   #SA to turn on the pasture area bound
sav['bnd_propn_dams_mated_og1'] = np.full((pinp.len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #proportion of dams mated
sav['bnd_drys_sold'] = '-'   #SA to force drys to be sold
sav['bnd_drys_retained'] = '-'   #SA to force drys to be retained
sav['bnd_sale_twice_dry_inc'] = '-'   #SA to include the the bound which forces twice dry dams to be sold
sav['bnd_twice_dry_propn'] = '-'   #SA to change twice dry dam proportion
sav['bnd_lower_dam_inc'] = '-'   #control if dam lower bound is on.
sav['bnd_upper_dam_inc'] = '-'   #control if dam upper bound is on.
sav['bnd_total_dams_scanned'] = '-'   #total dams scanned (summed over all dvps) - this also controls if bound is on.
sav['bnd_propn_dam5_retained'] = '-'   #propn of 5yo dams retained - this also controls if bound is on.
sav['bnd_sr_t'] = np.full(pinp.sheep['i_sr_constraint_t'].shape, '-', dtype=object)   #SA to fix stocking rate
sav['bnd_min_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age wether can be sold
sav['bnd_max_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
sav['bnd_min_sale_age_female_g1'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age a dam can be sold - BBT offspring can be sold but BBT dams can't (because they are BB)
sav['bnd_min_sale_age_female_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age a female can be sold - used to bound prog & offs

##pasture
sav['pas_inc'] = np.full_like(pinp.general['pas_inc'], '-', dtype=object) #SA value for pastures included mask

##Stock
sav['fec_inc'] = '-'    #SA to store FEC report values
sav['lw_inc'] = '-'     #SA to store LW report values
sav['ffcfw_inc'] = '-'  #SA to store FFCFW report values
sav['onhand_mort_p_inc'] = '-'  #SA to store onhand report values
sav['mort_inc'] = '-'  #SA to store mort report values
sav['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
sav['eqn_used_g0_q1p7'] = np.full(uinp.sheep['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g1_q1p7'] = np.full(uinp.sheep['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g2_q1p7'] = np.full(uinp.sheep['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g3_q1p7'] = np.full(uinp.sheep['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['TOL_inc']          = np.full(pinp.sheep['i_mask_i'].shape, '-', dtype=object)      # SA value for the inclusion of each TOL
sav['g3_included']      = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)      # SA value for the inclusion of each offspring genotype
sav['genotype']         = np.full(pinp.sheep['a_c2_c0'].shape, '-', dtype=object)       # this is the selection of the genotypes of the sires for B, M & T
sav['scan_og1']         = np.full(pinp.sheep['i_scan_og1'].shape, '-', dtype=object)    # SA value for the scanning management option
sav['woolp_mpg_percentile'] = '-'               #sa value for the wool price percentile
sav['woolp_mpg'] = '-'                          # sa value for wool price at std micron
sav['woolp_fdprem_percentile'] = '-'            # sa value for fd premium percentile (premium received by fd compared to std)
sav['woolp_fdprem'] = '-'                       # sa value for fd premium
sav['salep_percentile'] = '-'                   #Value for percentile for all sale grids
sav['salep_max'] = '-'                          #max sale price in grid
len_max_w1 = sinp.structuralsa['i_w_start_len1'] * len(sinp.structuralsa['i_nut_spread_n1']) ** (
        len(sinp.stock['i_fixed_fvp_mask_dams'])+len(sinp.structuralsa['i_fvp_mask_dams'])) #the max size of w if all n and fvps included.
len_max_w3 = sinp.structuralsa['i_w_start_len3'] * len(sinp.structuralsa['i_nut_spread_n3']) ** len(sinp.structuralsa['i_fvp_mask_offs']) #the max size of w if all n and fvps included.
sav['nut_mask_dams_owi'] = np.full((pinp.sheep['i_o_len'], len_max_w1, pinp.sheep['i_i_len']), '-', dtype=object)    #masks the nutrition options available eg high low high - the options selected are available for each starting weight. This array is cut down in the code to the correct w len.
sav['nut_mask_offs_swix'] = np.full((pinp.sheep['i_s_len'], len_max_w3, pinp.sheep['i_i_len'], pinp.sheep['i_x_len']), '-', dtype=object)   #masks the nutrition options available eg high low high - the options selected are available for each starting weight. This array is cut down in the code to the correct w len.
sav['nut_spread_n1'] = np.full(sinp.structuralsa['i_nut_spread_n1'].shape, '-', dtype=object)      #nut spread dams
sav['nut_spread_n3'] = np.full(sinp.structuralsa['i_nut_spread_n3'].shape, '-', dtype=object)      #nut spread offs
sav['n_fs_dams'] = '-'      #nut options dams
sav['n_fs_offs'] = '-'      #nut options offs
sav['n_initial_lw_dams'] = '-'      #number of initial lws dams - note with the current code this can only be 2 or 3
sav['adjp_lw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_lw_initial_w1'].shape, '-', dtype=object)      #initial lw adjustment dams
sav['mask_fvp_dams'] = np.full(sinp.structuralsa['i_fvp_mask_dams'].shape, '-', dtype=object)      #SA to mask changeable fvps.
sav['fvp_is_dvp_dams'] = np.full(sinp.structuralsa['i_dvp_mask_f1'].shape, '-', dtype=object)      #SA to control if changeable fvp is a dvp.
sav['mask_fvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to mask changeable fvps.
sav['fvp_is_dvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to control if changeable fvp is a dvp.
sav['r1_izg1'] = np.full(pinp.sheep['ia_r1_zig1'].shape, '-', dtype=object)   #SA to change the base feed option selected for dams
sav['r2_isk2g1'] = np.full(pinp.sheep['ia_r2_sk2ig1'].shape, '-', dtype=object)   #SA to Change the selected feed adjustments selected for the k2 axis (LSLN) for dams
sav['r1_izg3'] = np.full(pinp.sheep['ia_r1_zig3'].shape, '-', dtype=object)   #SA to change the base feed option selected for offspring

##stock parameters
sav['srw_c2'] = np.full(uinp.parameters['i_srw_c2'].shape, '-', dtype=object)  #SA value for srw of each c2 genotype.
sav['cl0_c2'] = np.full(uinp.parameters['i_cl0_c2'].shape, '-', dtype=object)  #SA value for litter size genotype params.
