# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John


A module of 2 functions that is called by FeedTest.py to calculate
    1. the NV for increments of the feed supply inputs
    2. the dates of the FVPs (for graphing).
The structure is based on the Sheep Generator but only calculates the feed inputs (NV) for dams
 (because sheep class doesn't alter the values i.e. no effect of age or gender on relative intake, although small impact for cattle of bite size)

The FVPs are calculated for both dams and offspring.

Last updated with StockGen code on 14Mar21
To update:
    Annotate StockGen code and locate lines or sections that have been altered more recently.
    Check each of the functions in FeedSupplyGenerator and replace if the code exists (or should now exist)
    Then starting at the bottom comment out any of the variables that are not used in the calculation of the variables that are returned


"""


"""
import functions from other modules
"""
import numpy as np

import Functions as fun
import FeedsupplyFunctions as fsfun
import PropertyInputs as pinp
import StockFunctions as sfun
import UniversalInputs as uinp
import StructuralInputs as sinp
import Periods as per


def feed_generator():
    """
    A function that generates the nutritive value (NV) for each feed supply value in each feed period.

    Called after the sensitivity variables have been updated.
    It populates the arrays by looping through the feed periods

    Returns
    -------
    None.
    """

    print("starting feed supply generator")

    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
#    w_pos = sinp.stock['i_w_pos']
#    x_pos = sinp.stock['i_x_pos']
#    y_pos = sinp.stock['i_y_pos']
    z_pos = sinp.stock['i_z_pos']


    ######################
    ##date               #
    ######################
    na=np.newaxis
    ## define the periods - default (dams and sires)
    n_sim_periods = 11  #number of feed periods + 1
    date_start_pz = per.f_feed_periods()[0:-1].astype('datetime64') #remove last date because that is the end date of the last period (not required)
    date_start_p = date_start_pz[:, 0] # to take only the first slice of the z axis. The Feed supply calculator.xlsx is not set to handle different seasons
    # date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(sinp.stock['i_p_pos']+1, 0)))


    ###################################
    ### axis len & index arrays       #
    ###################################
    ## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
    len_p = len(date_start_p)
    #len_n = 3
    len_w = 3001

    index_p = np.arange(len_p)
    index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(index_p, axis=tuple(range(sinp.stock['i_p_pos'] + 1, 0)))
    # index_n = np.arange(len_n)
    # index_nwzida0e0b0xyg = np.expand_dims(index_n, axis=tuple(range(uinp.structure['i_n_pos'] + 1, 0)))
    index_w = np.arange(len_w)
    index_wzida0e0b0xyg = np.expand_dims(index_w, axis=tuple(range(sinp.stock['i_w_pos'] + 1, 0)))

    ############################
    ### initialise arrays      #
    ############################
    '''only if assigned with a slice'''
    ##unique array shapes required to initialise arrays
    pw1 = (len_p, len_w)


    ##output variables for postprocessing & reporting
    dtype ='float64' #using 64 was getting slow

    ##dams
    ###arrays for postprocessing
    o_pi = np.zeros(pw1, dtype = dtype)
    o_mei = np.zeros(pw1, dtype = dtype)


    ###################################
    ### Create feedsupply             #
    ###################################

    # create a feed supply between 0 and 3 in 0.001 increments across the w axis.
    feedsupply = index_wzida0e0b0xyg / 1000 #  + index_nwzida0e0b0xyg
    ## make the last entry 4, which is ad lib supplement in confinement
    feedsupply[-1] = 4


    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from c to g
    ###production params
    srw_female_yg0, srw_female_yg1, srw_female_yg2, srw_female_yg3 = sfun.f1_c2g(uinp.parameters['i_srw_c2'], uinp.parameters['i_srw_y']) #srw of a female of the given genotype (this is the definition of the inputs)

    ###sim params
    ci_sire, ci_dams, ci_yatf, ci_offs = sfun.f1_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'])
#    ck_sire, ck_dams, ck_yatf, ck_offs = sfun.f1_c2g(uinp.parameters['i_ck_c2'], uinp.parameters['i_ck_y'], uinp.parameters['i_ck_pos'])
#    cl0_sire, cl0_dams, cl0_yatf, cl0_offs = sfun.f1_c2g(uinp.parameters['i_cl0_c2'], uinp.parameters['i_cl0_y'], uinp.parameters['i_cl0_pos'])
#    cl1_sire, cl1_dams, cl1_yatf, cl1_offs = sfun.f1_c2g(uinp.parameters['i_cl1_c2'], uinp.parameters['i_cl1_y'], uinp.parameters['i_cl1_pos'])
    cl_sire, cl_dams, cl_yatf, cl_offs = sfun.f1_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'])
#    cm_sire, cm_dams, cm_yatf, cm_offs = sfun.f1_c2g(uinp.parameters['i_cm_c2'], uinp.parameters['i_cm_y'], uinp.parameters['i_cm_pos'])
#    cn_sire, cn_dams, cn_yatf, cn_offs = sfun.f1_c2g(uinp.parameters['i_cn_c2'], uinp.parameters['i_cn_y'], uinp.parameters['i_cn_pos'])
#    cp_sire, cp_dams, cp_yatf, cp_offs = sfun.f1_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'])
    cr_sire, cr_dams, cr_yatf, cr_offs = sfun.f1_c2g(uinp.parameters['i_cr_c2'], uinp.parameters['i_cr_y'], uinp.parameters['i_cr_pos'])
#    crd_sire, crd_dams, crd_yatf, crd_offs = sfun.f1_c2g(uinp.parameters['i_crd_c2'], uinp.parameters['i_crd_y'], uinp.parameters['i_crd_pos'])
    cu0_sire, cu0_dams, cu0_yatf, cu0_offs = sfun.f1_c2g(uinp.parameters['i_cu0_c2'], uinp.parameters['i_cu0_y'], uinp.parameters['i_cu0_pos'])
#    cu1_sire, cu1_dams, cu1_yatf, cu1_offs = sfun.f1_c2g(uinp.parameters['i_cu1_c2'], uinp.parameters['i_cu1_y'], uinp.parameters['i_cu1_pos'])
#    cu2_sire, cu2_dams, cu2_yatf, cu2_offs = sfun.f1_c2g(uinp.parameters['i_cu2_c2'], uinp.parameters['i_cu2_y'], uinp.parameters['i_cu2_pos'])
#    cw_sire, cw_dams, cw_yatf, cw_offs = sfun.f1_c2g(uinp.parameters['i_cw_c2'], uinp.parameters['i_cw_y'], uinp.parameters['i_cw_pos'])
    cx_sire, cx_dams, cx_yatf, cx_offs = sfun.f1_c2g(uinp.parameters['i_cx_c2'], uinp.parameters['i_cx_y'], uinp.parameters['i_cx_pos'])
    ##pasture params
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp can't handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp can't handle object arrays)

    ###################################
    ###group independent              #  type(pinp.sheep['i_mask_z']).dtype
    ###################################
    ##legume proportion in each period
    legume_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_legume_p6z'],z_pos,source=0,dest=-1,left_pos2=p_pos,
                                                 right_pos2=z_pos)  # p6 axis converted to p axis later (association section)
    ##estimated foo and dmd for the feed periods (p6) periods
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_paststd_foo_p6zj0'],z_pos,move=True,source=0,
                                                       dest=2,
                                                       left_pos2=n_pos,right_pos2=z_pos,left_pos3=p_pos,
                                                       right_pos3=n_pos)  # p6 axis converted to p axis later (association section), axis order doesnt matter because sliced when used
    pasture_stage_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_pasture_stage_p6z'],z_pos,move=True,source=0,
                                                         dest=-1,
                                                         left_pos2=p_pos,
                                                         right_pos2=z_pos)  # p6 axis converted to p axis later (association section), z is treated later also
    ##foo for a region of measurement with pasture stage converted to hand shears and estimated height (for GrazFeed) - the z axis is also treated in this step
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg1, paststd_hf_p6a1e1b1j0wzida0e0b0xyg1 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg,
                                                                                     cr_dams,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    ##treat z axis (have to do it after adjusting foo)
    legume_p6a1e1b1nwzida0e0b0xyg = pinp.f_seasonal_inp(legume_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos)
    ##dmd
    paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_paststd_dmd_p6zj0'],z_pos,move=True,source=0,
                                                       dest=2,
                                                       left_pos2=n_pos,right_pos2=z_pos,left_pos3=p_pos,
                                                       right_pos3=n_pos)  # p6 axis converted to p axis later (association section), axis order doesnt matter because sliced when used
    paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = pinp.f_seasonal_inp(paststd_dmd_p6a1e1b1j0wzida0e0b0xyg,numpy=True,axis=z_pos)
    ##season type probability - prob and z mask are accounted for in f_season
#    i_season_propn_z = fun.f_expand(np.ones_like(pinp.general['i_season_propn_z']),z_pos)
#    season_propn_zida0e0b0xyg = pinp.f_seasonal_inp(i_season_propn_z,numpy=True,axis=z_pos)
    ##wind speed
#    ws_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_ws_p4'],p_pos)
    ##expected stocking density
#    density_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_density_p6z'],z_pos,source=0,dest=-1,
#                                                  left_pos2=p_pos, right_pos2=z_pos)  # p6 axis converted to p axis later (association section)
#    density_p6a1e1b1nwzida0e0b0xyg = pinp.f_seasonal_inp(density_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos).astype(int)
    ##nutrition adjustment for expected stocking density
#    density_nwzida0e0b0xyg1 = fun.f_expand(sinp.stock['i_density_n1'],n_pos)
#    density_nwzida0e0b0xyg3 = fun.f_expand(sinp.stock['i_density_n3'],n_pos)
    ##Calculation of rainfall distribution across the week - i_rain_distribution_p4p1 = how much rain falls on each day of the week sorted in order of quantity of rain. SO the most rain falls on the day with the highest rainfall.
    rain_p4a1e1b1nwzida0e0b0xygp1 = fun.f_expand(
        pinp.sheep['i_rain_p4'][...,na] * pinp.sheep['i_rain_distribution_p4p1'] * (7 / 30.4),p_pos - 1,
        right_pos=-1)  # -1 because p is -16 when p1 axis is included
    ##Mean daily temperature
    temp_ave_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_ave_p4'],p_pos)
    ##Mean daily maximum temperature
    temp_max_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_max_p4'],p_pos)
    ##Mean daily minimum temperature
    temp_min_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_min_p4'],p_pos)
    ##latitude



    ############################
    ### associations           #
    ############################
    ### Aim is to create an association array that points from period to opportunity
    ## The date_p array will be the start date for each period.
    ## A pointer (association) is required that points from the period to the previous or next lambing opportunity
    ## The Lambing opportunity is defined by the joining_date array (which has an 'o' axis)
    ## The previous lambing opportunity is the latest joining date that is less than the date at the end of the period (ie previous/current opportunity)
    ## # ('end of the period' so that if joining occurs during the period it is the previous
    ## The next lambing opportunity is the earliest joining date that is greater than or equal to the end date at the end of the period
    ##prev is just next - 1





    ##month of each period (0 - 11 not 1 -12 because this is association array)
    a_p4_p = date_start_p.astype('datetime64[M]').astype(int) % 12


    ############################
    ### apply associations     #
    ############################
    '''
    The association applied determines when the increment to the next opportunity will occur:
        eg if you use a_prev_joining the date in the p slice will increment at joining each time.
    
    '''
    ##feed period
    legume_pa1e1b1nwzida0e0b0xyg = legume_p6a1e1b1nwzida0e0b0xyg
    ##select which equation is used for the sheep sim functions for each period
    eqn_used_g1_q1p = uinp.sheep['i_eqn_used_g1_q1p7'][:, 0:1]
    ##convert foo and dmd for each feed period to each sim period
    ### the p axis is the p6 axis so no change required
    paststd_foo_pa1e1b1j0wzida0e0b0xyg = paststd_foo_p6a1e1b1j0wzida0e0b0xyg1
    paststd_dmd_pa1e1b1j0wzida0e0b0xyg = paststd_dmd_p6a1e1b1j0wzida0e0b0xyg
    paststd_hf_pa1e1b1j0wzida0e0b0xyg = paststd_hf_p6a1e1b1j0wzida0e0b0xyg1

    ##weather
    rain_pa1e1b1nwzida0e0b0xygp1 = rain_p4a1e1b1nwzida0e0b0xygp1[a_p4_p]
    temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_p4a1e1b1nwzida0e0b0xyg[a_p4_p]


    ###########################
    ##genotype calculations   #
    ###########################

    ###gender adjustment for srw
    srw_xyg1 = srw_female_yg1 * cx_dams[11, 1:2, ...]



    ############################
    ### Daily steps            #
    ############################
    ##definition for this is that the action eg weaning occurs at 12am on the given date. therefore is weaning occurs on day 150 the lambs are counted as weaned lambs on that day.
    ##This info determines the side with > or >=

    ##Impact of rainfall on 'cold' intake increment
    rain_intake_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp1 / ci_dams[18, ..., na]),  weights=np.array([1]), axis = -1)




    #####################################
    ##expand feedsupply for all w slices#   Note: This feedsupply is only the values from 0 to 4. Not related to feedsupply from StockGen
    #####################################
    ##add p axis to feedsupply
    feedsupplyw_pa1e1b1nwzida0e0b0xyg1 = feedsupply * (index_pa1e1b1nwzida0e0b0xyg >= 0)
    # ###apply association - give feedsupply singleton n axis and 81 slices in w axis
    # feedsupplyw_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg1,a_n_pa1e1b1nwzida0e0b0xyg1,axis=n_pos)


    ####################################
    ### initialise arrays for sim loop  # axis names not always track from now on because they change between p=0 and p=1
    ####################################
    ##dams
    temp_lc_dams = np.array([0.0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.


    ######################
    ### sim engine       #
    ######################


    ## Loop through each week of the simulation (p) for dams.
    for p in range(n_sim_periods-1):   #-1 because assigns to [p+1] for start values

        ###################
        ##dependent start #
        ###################
        ##dams
        ###Relative condition (start)
        rc_start_dams = np.array([1])
        ###Relative size (start) - dams & sires
        relsize_start_dams = np.array([1])
        ###PI Size factor (for cattle)
        zf_dams = np.maximum(1, 1 + cr_dams[7, ...] - relsize_start_dams)




        ##feed supply loop
        ##this loop is only required if a LW target is specified for the animals
        ##if there is a target then the loop needs to continue until
        ##the feed supply has converged on a value that generates a liveweight
        ##change close to the target
        ##The loop needs to execute at least once, then repeat if there
        ##is a target and the result is not close enough to the target


        ##potential intake
        eqn_group = 4
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                pi_dams = sfun.f_potential_intake_cs(ci_dams, cl_dams, srw_xyg1, relsize_start_dams, rc_start_dams, temp_lc_dams
                                                   , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                  , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg1[p])

        ###murdoch #todo function doesnt exist yet, add args when it is built
        eqn_system = 1 # mu = 1
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                pi_dams = sfun.f_potential_intake_mu(srw_xyg1)

        ##feedsupply - calculated after pi because pi required for intake_s
        foo_dams, hf_dams, dmd_dams, intake_s_dams, md_herb_dams  \
            = sfun.f1_feedsupply(feedsupplyw_pa1e1b1nwzida0e0b0xyg1[p] , paststd_foo_pa1e1b1j0wzida0e0b0xyg[p]
                                , paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], paststd_hf_pa1e1b1j0wzida0e0b0xyg[p], pi_dams)

        ##relative availability
        eqn_group = 5
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                ra_dams = fsfun.f_ra_cs(foo_dams, hf_dams, cr_dams, zf_dams)

        eqn_system = 1 # Murdoch = 1
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                ra_dams = fsfun.f_ra_mu(foo_dams, hf_dams, zf_dams, cu0_dams)


        ##relative quality/ingestibility
        eqn_group =  6
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                rq_dams = fsfun.f_rq_cs(dmd_dams, legume_pa1e1b1nwzida0e0b0xyg[p], cr_dams)

        ##intake
        ri_dams = fsfun.f_rel_intake(ra_dams, rq_dams, legume_pa1e1b1nwzida0e0b0xyg[p])
        mei_dams, mei_solid_dams, intake_f_dams, md_solid_dams, mei_propn_milk_dams, mei_propn_herb_dams, mei_propn_supp_dams  \
            = sfun.f_intake(pi_dams, ri_dams,  md_herb_dams, feedsupplyw_pa1e1b1nwzida0e0b0xyg1[p]
                            , intake_s_dams, pinp.sheep['i_md_supp'])



        ######################################
        #store postprocessing and report vars#
        ######################################
        ###dams
        ### Return the n & w axes with everything else slice 0 (because Pandas can only handle a 2D array)
        pi = pi_dams[0,0,0,0,:,0,0,0,0,0,0,0,0,0]
        mei = mei_solid_dams[0,0,0,0,:,0,0,0,0,0,0,0,0,0]
        ###store output variables for the post processing
        o_pi[p] = pi
        o_mei[p] = mei

    # calculate nutritive value of the diet
    r_nv = o_mei / o_pi
    # print (r_nv)

    return r_nv, np.squeeze(feedsupply)



def period_generator():
    '''A function that creates the generator periods and feed variation periods 
    that are used as part of creating the feedsupply inputs'''

    ######################
    ##background vars    #
    ######################
    na=np.newaxis
    a0_pos = sinp.stock['i_a0_pos']
#    a1_pos = sinp.stock['i_a1_pos']
#    b0_pos = sinp.stock['i_b0_pos']
#    b1_pos = sinp.stock['i_b1_pos']
    d_pos = sinp.stock['i_d_pos']
    e0_pos = sinp.stock['i_e0_pos']
    e1_pos = sinp.stock['i_e1_pos']
    i_pos = sinp.stock['i_i_pos']
#    k2_pos = sinp.stock['i_k2_pos']
#    k3_pos = sinp.stock['i_k3_pos']
#    k5_pos = sinp.stock['i_k5_pos']
#    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
#    w_pos = sinp.stock['i_w_pos']
    x_pos = sinp.stock['i_x_pos']
#    y_pos = sinp.stock['i_y_pos']
#    z_pos = sinp.stock['i_z_pos']

    ######################
    ##date               #
    ######################
    ## define the periods - default (dams and sires)
    sim_years = sinp.stock['i_age_max']
    sim_years_offs = min(sinp.stock['i_age_max_offs'], sim_years)
    n_sim_periods, date_start_p, date_end_p, p_index_p, step \
        = sfun.f1_sim_periods(pinp.sheep['i_startyear'], sinp.stock['i_sim_periods_year'], sim_years)
    date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(p_pos+1, 0)))
#    date_end_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_end_p, axis = tuple(range(p_pos+1, 0)))
#    p_index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(p_index_p, axis = tuple(range(p_pos+1, 0)))
    ## define the periods - offs - these make the p axis customisable for offs which means they can be smaller
    n_sim_periods_offs, offs_date_start_p, offs_date_end_p, p_index_offs_p, step \
        = sfun.f1_sim_periods(pinp.sheep['i_startyear'], sinp.stock['i_sim_periods_year'], sim_years_offs)
    date_start_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_start_p, axis = tuple(range(p_pos+1, 0)))
#    date_end_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_end_p, axis = tuple(range(p_pos+1, 0)))
#    p_index_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(p_index_offs_p, axis = tuple(range(p_pos+1, 0)))
#    mask_p_offs_p = p_index_p<=(n_sim_periods_offs-1)

    ###################################
    ## calculate masks                #
    ###################################
#    ##masks required for initialising arrays
#    mask_sire_inc_g0 = np.any(sinp.stock['i_mask_g0g3'] * pinp.sheep['i_g3_inc'], axis =1)
#    mask_dams_inc_g1 = np.any(sinp.stock['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis =1)
#    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis =1)
    ##o/d mask - if dob is after the end of the sim then it is masked out -  the mask is created before the date of birth is adjusted to the start of a period however it is adjusted to the start of the next period so the mask won't cut out a birth event that actually would occur, additionally this is the birth of the first however the matrix sees the birth of average animal which is also later therefore if anything the mask will leave in unnecessary o slices
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = sfun.f1_g2g(pinp.sheep['i_date_born1st_oig2'],'yatf', i_pos, swap=True,left_pos2=p_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]')
    mask_o_dams = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_d_offs = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_x = pinp.sheep['i_gender_propn_x']>0
    fvp_mask_dams = np.concatenate([sinp.stock['i_fixed_fvp_mask_dams'], sinp.structuralsa['i_fvp_mask_dams']])
    fvp_mask_offs = sinp.structuralsa['i_fvp_mask_offs']

    ###################################
    ### axis len                      #
    ###################################
    ## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
    if pinp.general['steady_state']:
        len_z = 1
    else:
        len_z = np.count_nonzero(pinp.general['i_mask_z'])
    len_p6 = len(per.f_feed_periods()) - 1 #-1 because the end feed period date is included


    ###################################
    ### index arrays                  #
    ###################################
    index_e = np.arange(np.max(pinp.sheep['i_join_cycles_ig1']))
    index_e1b1nwzida0e0b0xyg = fun.f_expand(index_e, e1_pos)
    index_e0b0xyg = fun.f_expand(index_e, e0_pos)

    ##output variables for postprocessing & reporting
#    dtype='float32' #using 64 was getting slow

    ################################################
    #  management, age, date, timing inputs inputs #
    ################################################
    ##gender propn yatf
#    gender_propn_xyg = fun.f_expand(pinp.sheep['i_gender_propn_x'], x_pos, condition=mask_x, axis=0).astype(dtype)
    ##join
    join_cycles_ida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_join_cycles_ig1'],'dams',i_pos)[pinp.sheep['i_mask_i'],...]
    ##lamb and lost
#    gbal_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_gbal_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
#    gbal_da0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_gbal_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied
    ##scanning
    scan_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_scan_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
#    scan_da0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_scan_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied
    ##post weaning management
#    wean_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_wean_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
    ##association between offspring and sire/dam (used to determine the wean age of sire and dams based on the inputted wean age of offs)
#    a_g0_g1 = sfun.f1_g2g(pinp.sheep['ia_g0_g1'],'dams')
#    a_g3_g0 = sfun.f1_g2g(pinp.sheep['ia_g3_g0'],'sire')  # the sire association (pure bred B, M & T) are all based on purebred B because there are no pure bred M & T inputs
    a_g3_g1 = sfun.f1_g2g(pinp.sheep['ia_g3_g1'],'dams')  # if BMT exist then BBM exist and they will be in slice 1, therefore the association value doesn't need to be adjusted for "prior exclusions"
    ##age weaning- used to calc wean date and also to calc p1 stuff, sire and dams have no active a0 slice therefore just take the first slice
    age_wean1st_a0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_age_wean_a0g3'],'offs',a0_pos).astype('timedelta64[D]')[pinp.sheep['i_mask_a']]
#    age_wean1st_e0b0xyg0 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g0],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
    age_wean1st_e0b0xyg1 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g1],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
    ##date first lamb is born - need to apply i mask to these inputs - make sure animals are born at beginning of gen period
#    date_born1st_ida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_born1st_ig0'],'sire',i_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos).astype('datetime64[D]')
    date_born1st_ida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_date_born1st_ig1'],'dams',i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2[mask_o_dams,...] #input read in in the mask section
    date_born1st_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_date_born1st_idg3'],'offs',d_pos, condition=pinp.sheep['i_mask_i']
                                           , axis=i_pos, condition2=mask_d_offs, axis2=d_pos).astype('datetime64[D]')
#    ##mating
#    sire_propn_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_sire_propn_oig1'],'dams', i_pos, swap=True,
#                                                   left_pos2=p_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'],
#                                                   axis=i_pos, condition2=mask_o_dams, axis2=p_pos)
#    sire_periods_p8g0 = sfun.f1_g2g(pinp.sheep['i_sire_periods_p8g0'], 'sire', condition=pinp.sheep['i_mask_p8'], axis=0)
#    sire_periods_g0p8 = np.swapaxes(sire_periods_p8g0, 0, 1) #can't swap in function above because g needs to be in pos-1
    ##Shearing date - set to be on the last day of a sim period
#    ###sire
#    date_shear_sida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg0'], 'sire', x_pos, swap=True
#                                          ,left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos
#                                          )[...,0:1,:,:].astype('datetime64[D]') #slice x axis for only female
#    mask_shear_g0 = np.max(date_shear_sida0e0b0xyg0<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
#    date_shear_sida0e0b0xyg0 = date_shear_sida0e0b0xyg0[mask_shear_g0]
#    ###dam
#    date_shear_sida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg1'],'dams',x_pos ,swap=True,left_pos2=i_pos,right_pos2=x_pos,
#                                          condition=pinp.sheep['i_mask_i'], axis=i_pos
#                                          )[...,1:2,:,:].astype('datetime64[D]') #slice x axis for only female
#    mask_shear_g1 = np.max(date_shear_sida0e0b0xyg1<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
#    date_shear_sida0e0b0xyg1 = date_shear_sida0e0b0xyg1[mask_shear_g1]
    ###off - shearing can't occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg3'],'offs',x_pos,swap=True,left_pos2=i_pos,right_pos2=x_pos,
                                          condition=pinp.sheep['i_mask_i'], axis=i_pos
                                          , condition2=mask_x, axis2=x_pos).astype('datetime64[D]')
    mask_shear_g3 = np.max(date_shear_sida0e0b0xyg3<=offs_date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg3 = date_shear_sida0e0b0xyg3[mask_shear_g3]


    ############
    #fvp inputs#
    ############
    fvp0_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp0_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp1_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp1_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp2_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp2_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)

    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from c to g
    ###production params
    ###sim params
    cf_sire, cf_dams, cf_yatf, cf_offs = sfun.f1_c2g(uinp.parameters['i_cf_c2'], uinp.parameters['i_cf_y'], uinp.parameters['i_cf_pos'])
#    cg_sire, cg_dams, cg_yatf, cg_offs = sfun.f1_c2g(uinp.parameters['i_cg_c2'], uinp.parameters['i_cg_y'], uinp.parameters['i_cg_pos'])
#    ch_sire, ch_dams, ch_yatf, ch_offs = sfun.f1_c2g(uinp.parameters['i_ch_c2'], uinp.parameters['i_ch_y'], uinp.parameters['i_ch_pos'])
#    ci_sire, ci_dams, ci_yatf, ci_offs = sfun.f1_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'])
#    ck_sire, ck_dams, ck_yatf, ck_offs = sfun.f1_c2g(uinp.parameters['i_ck_c2'], uinp.parameters['i_ck_y'], uinp.parameters['i_ck_pos'])
#    cl0_sire, cl0_dams, cl0_yatf, cl0_offs = sfun.f1_c2g(uinp.parameters['i_cl0_c2'], uinp.parameters['i_cl0_y'], uinp.parameters['i_cl0_pos'])
#    cl1_sire, cl1_dams, cl1_yatf, cl1_offs = sfun.f1_c2g(uinp.parameters['i_cl1_c2'], uinp.parameters['i_cl1_y'], uinp.parameters['i_cl1_pos'])
#    cl_sire, cl_dams, cl_yatf, cl_offs = sfun.f1_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'])
#    cm_sire, cm_dams, cm_yatf, cm_offs = sfun.f1_c2g(uinp.parameters['i_cm_c2'], uinp.parameters['i_cm_y'], uinp.parameters['i_cm_pos'])
#    cn_sire, cn_dams, cn_yatf, cn_offs = sfun.f1_c2g(uinp.parameters['i_cn_c2'], uinp.parameters['i_cn_y'], uinp.parameters['i_cn_pos'])
    cp_sire, cp_dams, cp_yatf, cp_offs = sfun.f1_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'])


    ########################################
    #adjust input to align with gen periods#
    ########################################
    # 1. Adjust date born (average) = start period
    # 2. Calc date born1st from slice 0 subtract 8 days
    # 3. Calc wean date
    # 4 adjust wean date to occur at start period
    # 5 calc adjusted wean age

    ##calc and adjust date born average of group - convert from date of first lamb born to average date born of lambs in the first cycle
#    ###sire
#    date_born_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
#    date_born_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_p, date_born_ida0e0b0xyg0, 0, 'left')
#    date_born_ida0e0b0xyg0 = date_start_p[date_born_idx_ida0e0b0xyg0]
    ###dams
    date_born_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg1=fun.f_next_prev_association(date_start_p, date_born_ida0e0b0xyg1, 0, 'left')
    date_born_ida0e0b0xyg1 = date_start_p[date_born_idx_ida0e0b0xyg1]
    ###yatf
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle. e_index is to account for ewe cycles.
    date_born_idx_oa1e1b1nwzida0e0b0xyg2 =fun.f_next_prev_association(date_start_p, date_born_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_born_idx_oa1e1b1nwzida0e0b0xyg2]
    ###offs
    date_born_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + (index_e0b0xyg + 0.5) * cf_offs[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_p, date_born_ida0e0b0xyg3, 0, 'left')
    date_born_ida0e0b0xyg3 = offs_date_start_p[date_born_idx_ida0e0b0xyg3]
    ##recalc date_born1st from the adjusted average birth date
#    date_born1st_ida0e0b0xyg0 = date_born_ida0e0b0xyg0 - 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')
    date_born1st_ida0e0b0xyg1 = date_born_ida0e0b0xyg1 - 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born_oa1e1b1nwzida0e0b0xyg2[:,:,0:1,...] - 0.5 * cf_yatf[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis
    date_born1st_ida0e0b0xyg3 = date_born_ida0e0b0xyg3[:,:,:,0:1,...] - 0.5 * cf_offs[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis

    ##calc wean date (weaning input is counting from the date of the first lamb)
#    date_weaned_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + age_wean1st_e0b0xyg0
    date_weaned_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + age_wean1st_e0b0xyg1
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + age_wean1st_a0e0b0xyg3 #use off wean age
    date_weaned_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + age_wean1st_a0e0b0xyg3
    ###adjust weaning to occur at the beginning of generator period and recalc wean age
#    ####sire
#    date_weaned_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg0, 0, 'left')
#    date_weaned_ida0e0b0xyg0 = date_start_p[date_weaned_idx_ida0e0b0xyg0]
#    age_wean1st_ida0e0b0xyg0 = date_weaned_ida0e0b0xyg0 - date_born1st_ida0e0b0xyg0
    ####dams
    date_weaned_idx_ida0e0b0xyg1=fun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg1, 0, 'left')
    date_weaned_ida0e0b0xyg1 = date_start_p[date_weaned_idx_ida0e0b0xyg1]
#    age_wean1st_ida0e0b0xyg1 = date_weaned_ida0e0b0xyg1 -date_born1st_ida0e0b0xyg1
    ####yatf - this is the same as offs except without the offs periods (offs potentially have less periods)
    date_weaned_idx_oa1e1b1nwzida0e0b0xyg2=fun.f_next_prev_association(date_start_p, date_weaned_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_weaned_idx_oa1e1b1nwzida0e0b0xyg2]
#    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = date_weaned_oa1e1b1nwzida0e0b0xyg2 - date_born1st_oa1e1b1nwzida0e0b0xyg2
    ####offs
    date_weaned_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_p, date_weaned_ida0e0b0xyg3, 0, 'left')
    date_weaned_ida0e0b0xyg3 = offs_date_start_p[date_weaned_idx_ida0e0b0xyg3]
    age_wean1st_ida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 - date_born1st_ida0e0b0xyg3

    ##Shearing date - set to be on the last day of a sim period
#    ###sire
#    idx_sida0e0b0xyg0 = fun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg0,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
#    date_shear_sa1e1b1nwzida0e0b0xyg0 = fun.f_expand(date_end_p[idx_sida0e0b0xyg0], p_pos, right_pos=i_pos)
    ###dam
#    idx_sida0e0b0xyg1 = fun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg1,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
#    date_shear_sa1e1b1nwzida0e0b0xyg1 = fun.f_expand(date_end_p[idx_sida0e0b0xyg1], p_pos, right_pos=i_pos)
    ###off - shearing can't occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = np.maximum(date_born1st_ida0e0b0xyg3 + age_wean1st_ida0e0b0xyg3, date_shear_sida0e0b0xyg3) #shearing must be after weaning.
    idx_sida0e0b0xyg3 = fun.f_next_prev_association(offs_date_end_p, date_shear_sida0e0b0xyg3,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg3 = fun.f_expand(offs_date_end_p[idx_sida0e0b0xyg3], p_pos, right_pos=i_pos)


    ############################
    ## calc for associations   #
    ############################
    ##date joined (when the rams go in)
    ##date joined (when the rams go in)
    date_joined_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 - cp_dams[1,...,0:1,:].astype('timedelta64[D]') #take slice 0 from y axis because cp1 is not affected by genetic merit
    ##expand feed periods over all the years of the sim so that an association between sim period can be made.
    feedperiods_p6z = per.f_feed_periods().astype('datetime64[D]')[:-1] #remove last date because that is the end date of the last period (not required)
    feedperiods_p6z = feedperiods_p6z + np.timedelta64(365,'D') * ((date_start_p[0].astype('datetime64[Y]').astype(int) + 1970 -1) - (feedperiods_p6z[0].astype('datetime64[Y]').astype(int) + 1970)) #this is to make sure the first sim period date is greater than the first feed period date.
    feedperiods_p6z = (feedperiods_p6z  + (np.arange(np.ceil(sim_years +1)) * np.timedelta64(365,'D') )[...,na,na]).reshape((-1, len_z)) #expand then ravel to return 1d array of the feed period dates expanded the length of the sim. +1 because feed periods start and finish mid yr so add one to ensure they go to the end of the sim.
    feedperiods_idx_p6z = fun.f_next_prev_association(date_start_p, feedperiods_p6z - np.timedelta64(step/2,'D'), 0, 'left') #get the nearest generator period (hence minus half a period
    feedperiods_p6z = date_start_p[feedperiods_idx_p6z]


    ###################################
    # Feed variation period calcs dams#
    ###################################
    ##fvp/dvp types
#    season_vtype1 = sinp.stock['i_fvp_type1'][0]
#    prejoin_vtype1 = sinp.stock['i_fvp_type1'][1]
#    condense_vtype1 = prejoin_vtype1 #currently for dams condensing must occur at prejoining, most of the code is flexible to handle different timing except the lw_distribution section.
#    scan_vtype1 = sinp.stock['i_fvp_type1'][2]
#    birth_vtype1 = sinp.stock['i_fvp_type1'][3]
#    wean_ftype1 = sinp.stock['i_fvp_type1'][4]
#    other_ftype1 = sinp.stock['i_fvp_type1'][5]

    ##beginning - first day of generator
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg[0:1]
    ##season start is the earliest of dry seeding start and earliest break of season.
    season_start = np.minimum(np.datetime64(pinp.crop['dry_seed_start']), np.min(per.f_feed_periods().astype(np.datetime64)[0,:]))
    startseason_y = season_start + (np.arange(np.ceil(sim_years)) * np.timedelta64(365,'D'))
    seasonstart_ya1e1b1nwzida0e0b0xyg = fun.f_expand(startseason_y, left_pos=p_pos)
    idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, seasonstart_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (eg break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
    seasonstart_ya1e1b1nwzida0e0b0xyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]
    ##early pregnancy fvp start - The pre-joining accumulation of the dams from the previous reproduction cycle - this date must correspond to the start date of period
    prejoining_approx_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 - sinp.stock['i_prejoin_offset'] #approx date of prejoining - in the next line of code prejoin date is adjusted to be the start of a sim period in which the approx date falls
    idx = np.searchsorted(date_start_p, prejoining_approx_oa1e1b1nwzida0e0b0xyg1, 'right') - 1 #gets the sim period index for the period that prejoining occurs (eg prejoining fvp starts at the beginning of the sim period when prejoining approx occurs), side=right so that if the date is already the start of a period it remains in that period.
    prejoining_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx]
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = prejoining_oa1e1b1nwzida0e0b0xyg1
    ##late pregnancy fvp start - Scanning if carried out, day 90 from joining (ram in) if not scanned.
    late_preg_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') + pinp.sheep['i_scan_day'][scan_oa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]')
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, late_preg_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when dams in late preg (eg late preg fvp starts at the beginning of the sim period when late preg occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ## lactation fvp start - average date of lambing (with e axis)
    lactation_date_oa1e1b1nwzida0e0b0xyg1 = date_born_oa1e1b1nwzida0e0b0xyg2
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, lactation_date_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when lactation starts (eg lactation fvp starts at the beginning of the sim period when lactation occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ##weaning
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = date_weaned_oa1e1b1nwzida0e0b0xyg2
    ##user defined fvp - rounded to nearest sim period
    fvp_other_yi = sinp.structuralsa['i_fvp4_date_i'].astype(np.datetime64) + np.arange(np.ceil(sim_years))[:,na] * np.timedelta64(365,'D')
    fvp_other_ya1e1b1nwzida0e0b0xyg = fun.f_expand(fvp_other_yi, left_pos=i_pos, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, fvp_other_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (eg break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([seasonstart_ya1e1b1nwzida0e0b0xyg.shape[1:],fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[1:], fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[1:]]) #create shape which has the max size, this is used for o array

    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_seasonstart_ya1e1b1nwzida0e0b0xyg = np.broadcast_to(seasonstart_ya1e1b1nwzida0e0b0xyg,(seasonstart_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape))
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = np.broadcast_to(fvp_other_start_ya1e1b1nwzida0e0b0xyg,(fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape))
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))

    ##create fvp type arrays. these are the same shape as the start arrays and are filled with the number corresponding to the fvp number
#    fvp_season_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_seasonstart_ya1e1b1nwzida0e0b0xyg.shape,season_vtype1)
#    fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape,prejoin_vtype1)
#    fvp_scan_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape,scan_vtype1)
#    fvp_birth_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape,birth_vtype1)
#    fvp_wean_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape,wean_ftype1)
#    fvp_other_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape,other_ftype1)
#    fvp_begin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape,condense_vtype1)
    ##stack & mask which dvps are included - this must be in the order as per the input mask
    fvp_date_all_f1 = np.array([fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,
                               fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1, fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,
                               fvp_birth_start_oa1e1b1nwzida0e0b0xyg1, fvp_seasonstart_ya1e1b1nwzida0e0b0xyg,
                                fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,
                               fvp_other_start_ya1e1b1nwzida0e0b0xyg], dtype=object)
#    fvp_type_all_f1 = np.array([fvp_begin_type_va1e1b1nwzida0e0b0xyg1,fvp_season_type_va1e1b1nwzida0e0b0xyg1,
#                               fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1, fvp_scan_type_va1e1b1nwzida0e0b0xyg1,
#                               fvp_birth_type_va1e1b1nwzida0e0b0xyg1, fvp_wean_type_va1e1b1nwzida0e0b0xyg1,
#                               fvp_other_type_va1e1b1nwzida0e0b0xyg1], dtype=object)
    fvp1_inc = np.concatenate([np.array([True]), fvp_mask_dams]) #True at start is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
    fvp_date_inc_f1 = fvp_date_all_f1[fvp1_inc]
#    fvp_type_inc_f1 = fvp_type_all_f1[fvp1_inc]
    fvp_start_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_date_inc_f1,axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_type_inc_f1,axis=0)
    ##mask any that occur before weaning (except the start fvp) and set to last date of generator and type to 0 so they are essentially ignored.
    mask = np.logical_and(fvp_start_fa1e1b1nwzida0e0b0xyg1 <= date_weaned_ida0e0b0xyg1, fvp_start_fa1e1b1nwzida0e0b0xyg1 > date_start_p[0])
    fvp_start_fa1e1b1nwzida0e0b0xyg1[mask] = date_start_p[-1]
    ##if more than one dvp on the last day of the generator dates must be offset by 1 because can't have multiple dvps on same date.
    mask = fvp_start_fa1e1b1nwzida0e0b0xyg1 == date_start_p[-1] #can't use the existing mask (above) in case there is an fvp on the last day of generator that we didn't manually put there.
    fvp_start_fa1e1b1nwzida0e0b0xyg1 = fvp_start_fa1e1b1nwzida0e0b0xyg1 - ((np.cumsum(mask, axis=0) - 1) * mask) #if multiple fvps are before weaning then offset their date by 1 day so they are not on the same date.
#    fvp_type_fa1e1b1nwzida0e0b0xyg1[mask] = condense_vtype1 #set to condense type to make sure extra dvps don't cause issues with masking or feed supply
    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg1, axis=0)
    fvp_date_start_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)

    ####################################
    # Feed variation period calcs offs #
    ####################################
    ##fvp/dvp types
#    season_vtype3=sinp.stock['i_fvp_type3'][0]
#    shear_vtype3 = sinp.stock['i_fvp_type3'][1]
#    condense_vtype3 = sinp.stock['i_condensefvp_type3']
#    inter1_vtype3 = sinp.stock['i_fvp_type3'][2]
#    inter2_vtype3 = sinp.stock['i_fvp_type3'][3]

    ##fvp's between weaning and first shearing - there will be 3 fvp's equally spaced between wean and first shearing (unless shearing occurs within 3 periods from weaning - if weaning and shearing are close the extra fvp are masked out in the stacking process below)
    ###b0
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg3[0:1]
    ###b1
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b1_start_ba1e1b1nwzida0e0b0xyg3, side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_ba1e1b1nwzida0e0b0xyg]
    ###b2
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + 2*(date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_ba1e1b1nwzida0e0b0xyg]
    ##season start is the earliest of dry seeding start and earliest break of season.
    startseason_y3 = season_start + (np.arange(np.ceil(sim_years_offs)) * np.timedelta64(365,'D'))
    seasonstart_ya1e1b1nwzida0e0b0xyg3 = fun.f_expand(startseason_y3, left_pos=p_pos)
    idx_ya1e1b1nwzida0e0b0xyg3 = np.searchsorted(offs_date_start_p, seasonstart_ya1e1b1nwzida0e0b0xyg3, 'right')-1 #gets the sim period index for the period when season breaks (eg break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
    seasonstart_ya1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_ya1e1b1nwzida0e0b0xyg3]
    ##fvp0 - date shearing plus 1 day because shearing is the last day of period
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + np.maximum(1, fvp0_offset_ida0e0b0xyg3) #plus 1 at least 1 because shearing is the last day of the period and the fvp should start after shearing
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_0_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period. side=right so that if the date is already the start of a period it remains in that period.
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]
    ##fvp1 - date shearing plus offset1 (this is the first day of sim period)
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + fvp1_offset_ida0e0b0xyg3
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_1_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]
    ##fvp2 - date shearing plus offset2 (this is the first day of sim period)
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + fvp2_offset_ida0e0b0xyg3
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_2_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[1:]
                               ,seasonstart_ya1e1b1nwzida0e0b0xyg3.shape[1:], fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[1:], fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[1:]]) #create shape which has the max size, this is used for o array
    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_seasonstart_ya1e1b1nwzida0e0b0xyg3 = np.broadcast_to(seasonstart_ya1e1b1nwzida0e0b0xyg3,(seasonstart_ya1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_0_start_sa1e1b1nwzida0e0b0xyg3,(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_1_start_sa1e1b1nwzida0e0b0xyg3,(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_2_start_sa1e1b1nwzida0e0b0xyg3,(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))

#    fvp_b0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape,0)
#    fvp_b1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape,1)
#    fvp_b2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape,2)
#    fvp_season_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_seasonstart_ya1e1b1nwzida0e0b0xyg3.shape,season_vtype3)
#    fvp_0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape,shear_vtype3)
#    fvp_1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape,inter1_vtype3)
#    fvp_2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape,inter2_vtype3)

    ##stack & mask which dvps are included - this must be in the order as per the input mask
    fvp_date_all_f3 = np.array([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,
                               fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,
                               fvp_0_start_sa1e1b1nwzida0e0b0xyg3, fvp_1_start_sa1e1b1nwzida0e0b0xyg3,
                               fvp_2_start_sa1e1b1nwzida0e0b0xyg3, fvp_seasonstart_ya1e1b1nwzida0e0b0xyg3], dtype=object)
#    fvp_type_all_f3 = np.array([fvp_b0_type_va1e1b1nwzida0e0b0xyg3, fvp_b1_type_va1e1b1nwzida0e0b0xyg3,
#                               fvp_b2_type_va1e1b1nwzida0e0b0xyg3, fvp_season_type_va1e1b1nwzida0e0b0xyg3,
#                               fvp_0_type_va1e1b1nwzida0e0b0xyg3, fvp_1_type_va1e1b1nwzida0e0b0xyg3,
#                               fvp_2_type_va1e1b1nwzida0e0b0xyg3], dtype=object)
    ###create the fvp mask. fvps are masked out depending on what the user has specified (the extra fvps at the start are always included but set to the start date of generator if weaning is within 3weeks of shearing).
    mask_initial_fvp = (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3) < ((step.astype('timedelta64[D]')+1)*3)  #true if not enough gap between weaning and shearing for extra dvps.
    fvp3_inc = np.concatenate([np.array([True, True, True, ]), fvp_mask_offs]) #Trues at start is to count for the extra fvp at the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
    fvp_date_inc_f3 = fvp_date_all_f3[fvp3_inc]
#    fvp_type_inc_f3 = fvp_type_all_f3[fvp3_inc]
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(fvp_date_inc_f3,axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(fvp_type_inc_f3,axis=0)
    ###if shearing is less than 3 sim periods after weaning then set the break fvp dates to the first date of the sim (so they arent used)
    mask_initial_fvp = np.broadcast_to(mask_initial_fvp, fvp_start_fa1e1b1nwzida0e0b0xyg3.shape).copy()
    mask_initial_fvp[3:, ...] = False #only a mask for the 3 extra dvps at the beginning (all the others will not be altered hence set mask to false)
    fvp_start_fa1e1b1nwzida0e0b0xyg3[mask_initial_fvp] = offs_date_start_p[0]
#    fvp_type_fa1e1b1nwzida0e0b0xyg3[mask_initial_fvp] = 0

    ##mask any that occur before weaning and set to last date of generator and type to 0 so they are essentially ignored.
    mask = np.logical_and(fvp_start_fa1e1b1nwzida0e0b0xyg3 <= date_weaned_ida0e0b0xyg3, fvp_start_fa1e1b1nwzida0e0b0xyg3 > offs_date_start_p[0])
    fvp_start_fa1e1b1nwzida0e0b0xyg3[mask] = offs_date_start_p[-1]
    ###if more than one dvp on the last day of the generator dates must be offset by 1 because can't have multiple dvps on same date.
    mask = fvp_start_fa1e1b1nwzida0e0b0xyg3 == offs_date_start_p[-1] #can't use the existing mask (above) in case there is an fvp on the last day of generator that we didn't manually put there.
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = fvp_start_fa1e1b1nwzida0e0b0xyg3 - ((np.cumsum(mask, axis=0) - 1) * mask) #if multiple fvps are before weaning then offset their date by 1 day so they are not on the same date.
#    fvp_type_fa1e1b1nwzida0e0b0xyg3[mask] = condense_vtype3 #set to condense type to make sure extra dvps don't cause issues with masking or feed supply

    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg3, axis=0)
    fvp_date_start_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)

    ############################
    ### associations           #
    ############################
    ### Aim is to create an association array that points from period to opportunity
    ## The date_p array will be the start date for each period.
    ## A pointer (association) is required that points from the period to the previous or next lambing opportunity
    ## The Lambing opportunity is defined by the joining_date array (which has an 'o' axis)
    ## The previous lambing opportunity is the latest joining date that is less than the date at the end of the period (ie previous/current opportunity)
    ## # ('end of the period' so that if joining occurs during the period it is the previous
    ## The next lambing opportunity is the earliest joining date that is greater than or equal to the end date at the end of the period
    ##prev is just next - 1

    ##Feed period for each generator period
    a_p6_pz = np.apply_along_axis(fun.f_next_prev_association, 0, feedperiods_p6z, date_end_p, 1,'right') % len_p6 #% 10 required to convert association back to only the number of feed periods
#    a_p6_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_p6_pz,z_pos,left_pos2=p_pos,right_pos2=z_pos)


    ##feed variation period
#    a_fvp_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_date_start_fa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
#    a_fvp_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_date_start_fa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')

    return date_start_p, fvp_date_start_fa1e1b1nwzida0e0b0xyg1, fvp_date_start_fa1e1b1nwzida0e0b0xyg3, a_p6_pz