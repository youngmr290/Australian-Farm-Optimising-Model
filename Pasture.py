'''
Pasture documentation can be found in PastureFunctions.py
'''

'''
import functions from other modules
'''
# import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Functions as fun
import FeedsupplyFunctions as fsfun
import Periods as per
import Sensitivity as sen
import PastureFunctions as pfun

#1. todo add labour required for feed budgeting. Inputs are currently in the sheep sheet of Property.xls
#2. todo Will need to add the foo reduction in the current year for manipulated pasture and a germination reduction in the following year.

def f_pasture(params, r_vals, nv):
    ######################
    ##background vars    #
    ######################
    na = np.newaxis

    ########################
    ##nv stuff             #
    ########################
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement period is included the last nv pool is confinment.

    ########################
    ##phases               #
    ########################
    ## read the rotation phases information from inputs
    # phase_len       = sinp.general['phase_len']
    phases_rotn_df  = sinp.f_phases()
    pasture_sets    = sinp.landuse['pasture_sets']
    pastures        = sinp.general['pastures'][pinp.general['pas_inc']]

    ########################
    ##masks                #
    ########################
    ##lmu
    lmu_mask_l = pinp.general['lmu_area'].squeeze().values>0

    ########################
    ##constants required   #
    ########################
    ## define some parameters required to size arrays.
    n_feed_pools    = len_nv
    n_dry_groups    = len(sinp.general['dry_groups'])           # Low & high quality groups for dry feed
    n_grazing_int   = len(sinp.general['grazing_int'])          # grazing intensity in the growth/grazing activities
    n_foo_levels    = len(sinp.general['foo_levels'])           # Low, medium & high FOO level in the growth/grazing activities
    n_feed_periods  = len(per.f_feed_periods()) - 1
    n_lmu           = np.count_nonzero(pinp.general['lmu_area'])
    n_phases_rotn   = len(phases_rotn_df.index)
    n_pasture_types = len(pastures)   #^ need to sort timing of the definition of pastures
    # n_total_seasons = len(pinp.general['i_mask_z']) #used to reshape inputs
    if pinp.general['steady_state']:
        n_season_types = 1
    else:
        n_season_types = np.count_nonzero(pinp.general['i_mask_z'])

    index_f = np.arange(n_feed_periods)

    ## indexes required for advanced indexing
    l_idx = np.arange(n_lmu)
    z_idx = np.arange(n_season_types)
    t_idx = np.arange(n_pasture_types)
    r_idx = np.arange(n_phases_rotn)


    arable_l = pinp.crop['arable'].squeeze().values[lmu_mask_l]
    length_fz  = np.array(per.f_feed_periods(option=1),dtype='float64')
    feed_period_dates_fz = fun.f_baseyr(per.f_feed_periods()).astype('datetime64[D]') #feed periods are all date to the base yr (eg 2019) - this is required for some of the allocation formulas

    # fgop6lt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dgop6lzt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # fdp6t    = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
    fp6zt    = (n_feed_pools, n_feed_periods, n_season_types, n_pasture_types)
    # dp6t     = (n_dry_groups, n_feed_periods, n_pasture_types)
    # gop6lzt  = (n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # gop6t    = (n_grazing_int, n_foo_levels, n_feed_periods, n_pasture_types)
    gp6t     = (n_grazing_int, n_feed_periods, n_pasture_types)
    gt      = (n_grazing_int, n_pasture_types)
    op6lzt   = (n_foo_levels, n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    # dp6lrt   = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    dp6lrzt  = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    p6lrt    = (n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    p6lrzt   = (n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    # p6rt     = (n_feed_periods, n_phases_rotn, n_pasture_types)
    rt      = (n_phases_rotn, n_pasture_types)
    p6lt     = (n_feed_periods, n_lmu, n_pasture_types)
    p6lzt    = (n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    lt      = (n_lmu, n_pasture_types)
    lzt     = (n_lmu, n_season_types, n_pasture_types)
    p6t      = (n_feed_periods, n_pasture_types)
    p6zt     = (n_feed_periods, n_season_types, n_pasture_types)
    zt      = (n_season_types, n_pasture_types)
    # t       = (n_pasture_types)

    ## define the vessels that will store the input data that require pre-defining
    ### all need pre-defining because inputs are in separate pasture type arrays
    i_phase_germ_dict = dict()
    i_grn_senesce_daily_p6t      = np.zeros(p6t,  dtype = 'float64')              # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_p6zt       = np.zeros(p6zt,  dtype = 'float64')             # proportion of green feed that senesces in period (due to a water deficit or completing life cycle)
    dry_decay_daily_p6zt         = np.zeros(p6zt,  dtype = 'float64')             # daily decline in dry foo in each period
    i_end_of_gs_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_exists_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_decay_t               = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    # i_me_maintenance_vp6t        = np.zeros(vp6t,  dtype = 'float64')     # M/D level for target LW pattern
    c_pgr_gi_scalar_gp6t         = np.zeros(gp6t,  dtype = 'float64')     # pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_graze_propn_gt        = np.zeros(gt, dtype ='float64')        # proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_op6lzt             = np.zeros(op6lzt, dtype = 'float64')    # FOO level     for the FOO/growth/grazing variables.
    i_fxg_pgr_op6lzt             = np.zeros(op6lzt, dtype = 'float64')    # PGR level     for the FOO/growth/grazing variables.
    c_fxg_a_op6lzt               = np.zeros(op6lzt, dtype = 'float64')    # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_op6lzt               = np.zeros(op6lzt, dtype = 'float64')    # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_op6lt               = np.zeros(op6lt, dtype = 'float64')     # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_op6lt               = np.zeros(op6lt, dtype = 'float64')     # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_p6lzt              = np.zeros(p6lzt, dtype = 'float64') # green pasture digestibility in each period, LMU, season & pasture type.
    i_poc_intake_daily_p6lt      = np.zeros(p6lt, dtype = 'float64')  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_p6lt      = np.zeros(p6lt, dtype = 'float64')  # minimum foo at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lzt           = np.zeros(lzt,  dtype = 'float64') # scale the germination levels for each lmu
    i_restock_fooscalar_lt      = np.zeros(lt,  dtype = 'float64')  # scalar for FOO between LMUs when pastures are restocked after reseeding

    i_me_eff_gainlose_p6t        = np.zeros(p6t,  dtype = 'float64')  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_p6t          = np.zeros(p6t,  dtype = 'float64')  # green pasture trampling in each feed period as proportion of intake.
    i_dry_trampling_p6t          = np.zeros(p6t,  dtype = 'float64')  # dry pasture trampling   in each feed period as proportion of intake.
    i_base_p6t                   = np.zeros(p6t,  dtype = 'float64')  # lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_p6t     = np.zeros(p6t,  dtype = 'float64')  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_p6t          = np.zeros(p6t,  dtype = 'float64')  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_p6zt  = np.zeros(p6zt,  dtype = 'float64') # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_p6zt           = np.zeros(p6zt,  dtype = 'float64') # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_p6zt         = np.zeros(p6zt,  dtype = 'float64') # range in digestibility of dry feed if it is not grazed
    i_dry_foo_high_p6zt          = np.zeros(p6zt,  dtype = 'float64') # expected foo for the dry pasture in the high quality pool
    dry_decay_period_p6zt        = np.zeros(p6zt,  dtype = 'float64') # decline in dry foo for each period
    mask_dryfeed_exists_p6zt     = np.zeros(p6zt,  dtype = bool)      # mask for period when dry feed exists
    mask_greenfeed_exists_p6zt   = np.zeros(p6zt,  dtype = bool)      # mask for period when green feed exists
    i_germ_scalar_p6zt           = np.zeros(p6zt,  dtype = 'float64') # allocate the total germination between feed periods
    i_grn_cp_p6t                 = np.zeros(p6t,  dtype = 'float64')  # crude protein content of green feed
    i_dry_cp_p6t                 = np.zeros(p6t,  dtype = 'float64')  # crude protein content of dry feed
    i_poc_dmd_p6t                = np.zeros(p6t,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_p6t                = np.zeros(p6t,  dtype = 'float64')  # foo of pasture consumed on crop paddocks
    # grn_senesce_startfoo_p6t     = np.zeros(p6t,  dtype = 'float64')  # proportion of the FOO at the start of the period that senesces during the period
    # grn_senesce_pgrcons_p6t      = np.zeros(p6t,  dtype = 'float64')  # proportion of the (total or average daily) PGR that senesces during the period (consumption leads to a reduction in senescence)

    i_reseeding_date_start_zt   = np.zeros(zt, dtype = 'datetime64[D]')         # start date of seeding this pasture type
    i_reseeding_date_end_zt     = np.zeros(zt, dtype = 'datetime64[D]')         # end date of the pasture seeding window for this pasture type
    i_destock_date_zt           = np.zeros(zt, dtype = 'datetime64[D]')         # date of destocking this pasture type prior to reseeding
    i_destock_foo_zt            = np.zeros(zt, dtype = 'float64')               # kg of FOO that was not grazed prior to destocking for spraying prior to reseeding pasture (if spring sown)
    i_restock_date_zt           = np.zeros(zt, dtype = 'datetime64[D]')         # date of first grazing of reseeded pasture
    i_restock_foo_arable_t      = np.zeros(n_pasture_types, dtype = 'float64')  # FOO at restocking on the arable area of the resown pastures
    # reseeding_machperiod_t      = np.zeros(n_pasture_types, dtype = 'float64')  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
    i_germination_std_zt        = np.zeros(zt, dtype = 'float64')               # standard germination level for the standard soil type in a continuous pasture rotation
    # i_ri_foo_t                  = np.zeros(n_pasture_types, dtype = 'float64')  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    # poc_days_of_grazing_t       = np.zeros(n_pasture_types, dtype = 'float64')  # number of days after the pasture break that (moist) seeding can begin
    i_legume_zt                 = np.zeros(zt, dtype = 'float64')               # proportion of legume in the sward
    i_restock_grn_propn_t       = np.zeros(n_pasture_types, dtype = 'float64')  # Proportion of the FOO that is green when pastures are restocked after reseeding
    i_nv_maintenance_t         = np.zeros(n_pasture_types, dtype = 'float64')  # approximate nutritive value for maintenance (NV = M/D * relative intake)

#    germination_p6lrzt           = np.zeros(p6lrzt,  dtype = 'float64')  # germination for each rotation phase (kg/ha)
    # foo_dry_reseeding_dp6lrzt    = np.zeros(dp6lrzt, dtype = 'float64')  # dry FOO adjustment allocated to the low & high quality dry feed pools (kg/ha)
    # dry_removal_t_p6t            = np.zeros(p6t,   dtype = 'float64')  # Total DM removal from the tonne consumed (includes trampling)

    ### define the array that links rotation phase and pasture type
    pasture_rt                  = np.zeros(rt, dtype = 'float64')


    ## create numpy index for param dicts ^creating indexes is a bit slow
    ### the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
    keys_d  = np.asarray(sinp.general['dry_groups'])
    keys_f  = np.array(['nv{0}' .format(i) for i in range(len_nv)])
    keys_p6  = pinp.period['i_fp_idx']
    keys_g  = np.asarray(sinp.general['grazing_int'])
    keys_l  = np.array(pinp.general['lmu_area'].index[lmu_mask_l]).astype('str')    # lmu index description
    keys_o  = np.asarray(sinp.general['foo_levels'])
    keys_p5  = np.array(per.p_date2_df().index).astype('str')
    keys_r  = np.array(phases_rotn_df.index).astype('str')
    keys_t  = np.asarray(pastures)                                      # pasture type index description
    keys_k  = np.asarray(list(sinp.landuse['All']))                     #landuse
    keys_z  = pinp.f_keys_z()

    ### plrk
    arrays=[keys_p5, keys_l, keys_r, keys_k]
    index_plrk=fun.cartesian_product_simple_transpose(arrays)
    index_plrk=tuple(map(tuple, index_plrk)) #create a tuple rather than a list because tuples are faster

    ### rt
    arrays=[keys_r, keys_t]
    index_rt=fun.cartesian_product_simple_transpose(arrays)
    index_rt=tuple(map(tuple, index_rt)) #create a tuple rather than a list because tuples are faster

    ### p6lrt
    arrays=[keys_p6, keys_l, keys_r, keys_t]
    index_p6lrt=fun.cartesian_product_simple_transpose(arrays)
    index_p6lrt=tuple(map(tuple, index_p6lrt)) #create a tuple rather than a list because tuples are faster

    ### op6lt
    arrays=[keys_o, keys_p6, keys_l, keys_t]
    index_op6lt=fun.cartesian_product_simple_transpose(arrays)
    index_op6lt=tuple(map(tuple, index_op6lt)) #create a tuple rather than a list because tuples are faster

    ### gop6lt
    arrays=[keys_g, keys_o, keys_p6, keys_l, keys_t]
    index_gop6lt=fun.cartesian_product_simple_transpose(arrays)
    index_gop6lt=tuple(map(tuple, index_gop6lt)) #create a tuple rather than a list because tuples are faster

    ### fgop6lt
    arrays=[keys_f, keys_g, keys_o, keys_p6, keys_l, keys_t]
    index_fgop6lt=fun.cartesian_product_simple_transpose(arrays)
    index_fgop6lt=tuple(map(tuple, index_fgop6lt)) #create a tuple rather than a list because tuples are faster

    ### dgop6lt
    arrays=[keys_d, keys_g, keys_o, keys_p6, keys_l, keys_t]
    index_dgop6lt=fun.cartesian_product_simple_transpose(arrays)
    index_dgop6lt=tuple(map(tuple, index_dgop6lt)) #create a tuple rather than a list because tuples are faster

    ### dp6lrt
    arrays=[keys_d, keys_p6, keys_l, keys_r, keys_t]
    index_dp6lrt=fun.cartesian_product_simple_transpose(arrays)
    index_dp6lrt=tuple(map(tuple, index_dp6lrt)) #create a tuple rather than a list because tuples are faster

    ### fdp6t
    arrays=[keys_f, keys_d, keys_p6, keys_t]
    index_fdp6t=fun.cartesian_product_simple_transpose(arrays)
    index_fdp6t=tuple(map(tuple, index_fdp6t)) #create a tuple rather than a list because tuples are faster

    ### dp6t
    arrays=[keys_d, keys_p6, keys_t]
    index_dp6t=fun.cartesian_product_simple_transpose(arrays)
    index_dp6t=tuple(map(tuple, index_dp6t)) #create a tuple rather than a list because tuples are faster

    ### fp6
    arrays=[keys_f, keys_p6]
    index_fp6=fun.cartesian_product_simple_transpose(arrays)
    index_fp6=tuple(map(tuple, index_fp6)) #create a tuple rather than a list because tuples are faster

    ### p6l
    arrays=[keys_p6, keys_l]
    index_p6l=fun.cartesian_product_simple_transpose(arrays)
    index_p6l=tuple(map(tuple, index_p6l)) #create a tuple rather than a list because tuples are faster

    ### p6t
    arrays=[keys_p6, keys_t]
    index_p6t=fun.cartesian_product_simple_transpose(arrays)
    index_p6t=tuple(map(tuple, index_p6t)) #create a tuple rather than a list because tuples are faster

    ###########
    #map_excel#
    ###########
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''

    ## map data from excel file into arrays
    ### loop through each pasture type
    for t, pasture in enumerate(pastures):
        exceldata = pinp.pasture_inputs[pasture]           # assign the pasture data to exceldata
        ## map the Excel data into the numpy arrays
        i_germination_std_zt[...,t]         = pinp.f_seasonal_inp(exceldata['GermStd'], numpy=True)
        # i_ri_foo_t[t]                       = exceldata['RIFOO']
        i_end_of_gs_zt[...,t]               = pinp.f_seasonal_inp(exceldata['EndGS'], numpy=True)
        i_dry_exists_zt[...,t]               = pinp.f_seasonal_inp(exceldata['i_dry_exists'], numpy=True)
        i_dry_decay_t[t]                    = exceldata['PastDecay']
        i_poc_intake_daily_p6lt[...,t]       = exceldata['POCCons'][:,lmu_mask_l]
        i_legume_zt[...,t]                  = pinp.f_seasonal_inp(exceldata['Legume'], numpy=True)
        i_restock_grn_propn_t[t]            = exceldata['FaG_PropnGrn']
        i_grn_dmd_senesce_redn_p6zt[...,t]   = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigRednSenesce'],0,1), numpy=True, axis=1)
        i_dry_dmd_ave_p6zt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryAve'],0,1), numpy=True, axis=1)
        i_dry_dmd_range_p6zt[...,t]          = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryRange'],0,1), numpy=True, axis=1)
        i_dry_foo_high_p6zt[...,t]           = pinp.f_seasonal_inp(np.swapaxes(exceldata['FOODryH'],0,1), numpy=True, axis=1)
        i_germ_scalar_p6zt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarFP'],0,1), numpy=True, axis=1)

        i_grn_cp_p6t[...,t]                  = exceldata['CPGrn']
        i_dry_cp_p6t[...,t]                  = exceldata['CPDry']
        i_poc_dmd_p6t[...,t]                 = exceldata['DigPOC']
        i_poc_foo_p6t[...,t]                 = exceldata['FOOPOC']
        i_germ_scalar_lzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarLMU'],0,1), numpy=True, axis=1)[lmu_mask_l,...]
        i_restock_fooscalar_lt[...,t]       = exceldata['FaG_LMU'][lmu_mask_l]  #todo may need a z axis

        i_lmu_conservation_p6lt[...,t]       = exceldata['ErosionLimit'][:, lmu_mask_l]

        i_reseeding_date_start_zt[...,t]    = pinp.f_seasonal_inp(exceldata['Date_Seeding'], numpy=True)
        i_reseeding_date_end_zt[...,t]      = pinp.f_seasonal_inp(exceldata['pas_seeding_end'], numpy=True)
        i_destock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_Destocking'], numpy=True)
        i_destock_foo_zt[...,t]             = pinp.f_seasonal_inp(exceldata['FOOatSeeding'], numpy=True) #ungrazed foo when destocked for reseeding
        i_restock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_ResownGrazing'], numpy=True)
        i_restock_foo_arable_t[t]           = exceldata['FOOatGrazing']

        i_grn_trampling_p6t[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_p6t[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_p6t[...,t]       = np.asfarray(exceldata['SenescePropn'])
        i_grn_senesce_eos_p6zt[...,t]        = pinp.f_seasonal_inp(np.asfarray(exceldata['SenesceEOS']), numpy=True, axis=1)
        i_base_p6t[...,t]                    = np.asfarray(exceldata['BaseLevelInput'])
        i_grn_dmd_declinefoo_p6t[...,t]      = np.asfarray(exceldata['DigDeclineFOO'])
        i_grn_dmd_range_p6t[...,t]           = np.asfarray(exceldata['DigSpread'])
        i_foo_graze_propn_gt[..., t]        = np.asfarray(exceldata['FOOGrazePropn'])
        #### impact of grazing intensity (at the other levels) on PGR during the period
        c_pgr_gi_scalar_gp6t[...,t]      = 1 - i_foo_graze_propn_gt[..., na, t] ** 2 * (1 - np.asfarray(exceldata['PGRScalarH']))

        i_fxg_foo_op6lzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowFOO'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_foo_op6lzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedFOO'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_me_eff_gainlose_p6t[...,t]     = exceldata['MaintenanceEff'][:,0]
        # i_me_maintenance_vp6t[...,t]     = exceldata['MaintenanceEff'].iloc[:,1:].to_numpy().T  # replaced by the nv_cutoff. Still used in PastureTest
        i_nv_maintenance_t[t]          = exceldata['MaintenanceNV']
        ## # i_fxg_foo_op6lt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_op6lzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_pgr_op6lzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_pgr_op6lzt[2,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_p6lzt[...,t]           = pinp.f_seasonal_inp(np.moveaxis(exceldata['DigGrn'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]  # numpy array of inputs for green pasture digestibility on each LMU.

        i_phase_germ_dict[pasture]      = pd.DataFrame(exceldata['GermPhases'])  #DataFrame with germ scalar and resown
        # i_phase_germ_dict[pasture].reset_index(inplace=True)                                # replace index read from Excel with numbers to match later merging
        # i_phase_germ_dict[pasture].columns.values[range(phase_len)] = [*range(phase_len)]   # replace the pasture columns read from Excel with numbers to match later merging

        ### define the link between rotation phase and pasture type while looping on pasture
        pasture_rt[:,t]                 = phases_rotn_df.iloc[:,-1].isin(pasture_sets[pasture])

    ##season inputs not required in t loop above
    harv_date_z         = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype(np.datetime64)
    i_pasture_stage_p6z = np.rint(pinp.f_seasonal_inp(np.moveaxis(pinp.sheep['i_pasture_stage_p6z'],0,-1), numpy=True, axis=-1)
                                  ).astype(int) #it would be better if z axis was treated after pas_stage has been used (like in stock.py) because it is used as an index. But there wasn't any way to do this without doubling up a lot of code. This is only a limitation in the weighted average version of model.
    ### pasture params used to convert foo for rel availability
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)

    ###create dry pasture exists mask - in the current structure dry pasture only exists after the growing season.
    mask_dryfeed_exists_p6zt[...] = index_f[:, na, na] >= i_dry_exists_zt   #mask periods when dry feed is available to livestock.
    mask_greenfeed_exists_p6zt[...] = index_f[:, na, na] <= i_end_of_gs_zt   #green exists in the period which is the end of growing season hence <=
    mask_dryfeed_exists_next_p6zt = np.roll(mask_dryfeed_exists_p6zt, shift=-1, axis=0)   #dry feed exists in the following feed period

    ### calculate dry_decay_period (used in reseeding and green&dry)
    ### dry_decay_daily is decay of dry foo at the start of the period that was either
    ### dry at the end of the last period or transferred in from senescence in the previous period.
    ### dry_decay_daily does not effect green feed that senesces during the current period.
    dry_decay_daily_p6zt[...] = i_dry_decay_t
    for t in range(n_pasture_types):
        for z in range(n_season_types):
            dry_decay_daily_p6zt[0:i_dry_exists_zt[z,t], z, t] = 1  #couldn't do this without loops - advanced indexing doesnt appear to work when taking multiple slices
    dry_decay_period_p6zt[...] = 1 - (1 - dry_decay_daily_p6zt) ** length_fz[...,na]
    ### allowance for the decay of dry feed in the days prior to being consumed
    ### because only the feed at the end of period is decayed by dry_decay_period_p6zt
    ### scales total removal to allow for an equal portion of the feed being grazed each day
    removal_scalar_dry_decay_daily_p6zt = fun.f_divide(1, (1 - dry_decay_daily_p6zt))
    removal_scalar_dry_decay_p6zt = ((1 - removal_scalar_dry_decay_daily_p6zt ** length_fz[..., na])
                                     / (1 - removal_scalar_dry_decay_daily_p6zt)
                                     / length_fz[..., na])
    ## dry, DM decline (high = low pools)
    ###dry transfer prov is the amount of dry feed that is transferred into the current period from the previous (1000 - decay)
    dry_transfer_prov_t_p6zt = 1000 * (1-dry_decay_period_p6zt) * mask_dryfeed_exists_next_p6zt #if no dry feed exists in the next period then we dont need the transfer prov DV.
    ###dry transfer required is the amount of dry feed required in the current period to transfer into the next period (1000 mask by dry exists)
    dry_transfer_req_t_p6zt = 1000 * mask_dryfeed_exists_p6zt #this parameter exists so that the constraint wont be built for fp when no dry feed exists.

    ###create equation coefficients for pgr = a+b*foo
    i_fxg_foo_op6lzt[2,...]  = 100000 #large number so that the np.searchsorted doesn't go above
    c_fxg_b_op6lzt[0,...] =  fun.f_divide(i_fxg_pgr_op6lzt[0,...], i_fxg_foo_op6lzt[0,...])
    c_fxg_b_op6lzt[1,...] =   fun.f_divide((i_fxg_pgr_op6lzt[1,...] - i_fxg_pgr_op6lzt[0,...])
                            , (i_fxg_foo_op6lzt[1,...] - i_fxg_foo_op6lzt[0,...]))
    c_fxg_b_op6lzt[2,...] =  0
    c_fxg_a_op6lzt[0,...] =  0
    c_fxg_a_op6lzt[1,...] =  i_fxg_pgr_op6lzt[0,...] - c_fxg_b_op6lzt[1,...] * i_fxg_foo_op6lzt[0,...]
    c_fxg_a_op6lzt[2,...] =  i_fxg_pgr_op6lzt[1,...] # because slope = 0

    ## proportion of start foo that senesces during the period, different formula than excel
    grn_senesce_startfoo_p6zt = 1 - ((1 - i_grn_senesce_daily_p6t[:,na,:]) **  length_fz[...,na])

    ## change of senescence over the period due to growth and consumption
    grn_senesce_pgrcons_p6zt = 1 - ((1 -(1 - i_grn_senesce_daily_p6t[:,na,:]) ** (length_fz[...,na]+1))
                                   /        i_grn_senesce_daily_p6t[:,na,:]-1) / length_fz[...,na]



    #################################################
    #Calculate germination and reseeding parameters #
    #################################################


    ## define instantiate arrays that are assigned in slices
    # na_erosion_p6lrt      = np.zeros(p6lrt,  dtype = 'float64')
    # na_phase_area_p6lrzt  = np.zeros(p6lrzt, dtype = 'float64')
    # grn_restock_foo_p6lzt = np.zeros(p6lzt,  dtype = 'float64')
    # dry_restock_foo_p6lzt = np.zeros(p6lzt,  dtype = 'float64')



    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df

    germination_p6lrzt, max_germination_flz, resown_rt = pfun.f_germination(i_germination_std_zt, i_germ_scalar_lzt
                                                                , i_germ_scalar_p6zt, pasture_rt, arable_l
                                                                , pastures, phase_germresow_df, i_phase_germ_dict, rt)

    foo_grn_reseeding_p6lrzt, foo_dry_reseeding_dp6lrzt, periods_destocked_p6zt = pfun.f_reseeding(
        i_destock_date_zt, i_restock_date_zt, i_destock_foo_zt, i_restock_grn_propn_t, resown_rt, feed_period_dates_fz
        , i_restock_fooscalar_lt, i_restock_foo_arable_t, dry_decay_period_p6zt, i_fxg_foo_op6lzt, c_fxg_a_op6lzt
        , c_fxg_b_op6lzt, i_grn_senesce_eos_p6zt, grn_senesce_startfoo_p6zt, grn_senesce_pgrcons_p6zt, max_germination_flz
        , length_fz, n_feed_periods, p6lrzt, p6zt, t_idx, z_idx, l_idx)

    ## sow param determination
    pas_sow_p5lrkz = pfun.f_pas_sow(i_reseeding_date_start_zt, i_reseeding_date_end_zt, resown_rt, arable_l, phases_rotn_df)

    ## area of green pasture being grazed and growing
    phase_area_p6lrzt = pfun.f1_green_area(resown_rt, pasture_rt, periods_destocked_p6zt, arable_l)

    ## erosion limit. The minimum FOO at the end of each period#
    erosion_p6lrt = pfun.f_erosion(i_lmu_conservation_p6lt, arable_l, pasture_rt)

    ## initialise numpy arrays used only in this method
    senesce_propn_dgop6lzt      = np.zeros(dgop6lzt, dtype = 'float64')
    nap_dp6lrzt                 = np.zeros(dp6lrzt,  dtype = 'float64')
    me_threshold_fp6zt          = np.zeros(fp6zt,    dtype = 'float64')   # the threshold for the nv pools which define the animals feed quality requirements

    ## create numpy array of threshold values from the nv dictionary
    ### note: v in pasture is f in StockGen and f in pasture is p6 in StockGen
    me_threshold_fp6zt[...] = np.swapaxes(nv['nv_cutoff_ave_p6f'], axis1=0, axis2=1)[...,na,na]
    ###threshold is the greater of the maintenance or the NV required in the pool because switching from one below
    ### maintenance feed to another that is further below maintenance doesn't affect average efficiency.
    me_threshold_fp6zt[me_threshold_fp6zt < i_nv_maintenance_t] = i_nv_maintenance_t

    ## FOO on the non-arable areas in crop paddocks is ungrazed FOO of pasture type 0 (annual), therefore calculate the profile based on the pasture type 0 values
    grn_foo_start_ungrazed_p6lzt, dry_foo_start_ungrazed_p6lzt = pfun.f1_calc_foo_profile(
        max_germination_flz[..., na], dry_decay_period_p6zt[..., 0:1], length_fz, i_fxg_foo_op6lzt[..., 0:1]
        , c_fxg_a_op6lzt[..., 0:1], c_fxg_b_op6lzt[..., 0:1], i_grn_senesce_eos_p6zt[..., 0:1], grn_senesce_startfoo_p6zt[..., 0:1]
        , grn_senesce_pgrcons_p6zt[..., 0:1])

    ### non arable pasture becomes available to graze at the beginning of the first harvest period
    # harvest_period  = fun.period_allocation(pinp.period['feed_periods']['date'], range(len(pinp.period['feed_periods'])), pinp.period['harv_date']) #use range(len()) to get the row number that harvest occurs has to be row number not index name because it is used to index numpy below
    harv_period_z, harv_proportion_z = fun.period_proportion_np(feed_period_dates_fz, harv_date_z)
    index = pd.MultiIndex.from_arrays([keys_p6[harv_period_z], keys_z])
    harvest_period_prop = pd.Series(harv_proportion_z, index=index).unstack()
    # params['p_harvest_period_prop']  = dict([(pinp.period['feed_periods'].index[harv_period_z], harv_proportion_z)])

    ### all pasture from na area goes into the Low pool (#1) because it is rank & low quality
    nap_dp6lrzt[0, harv_period_z, l_idx[:,na,na], r_idx[:,na], z_idx, 0] = (
                                             dry_foo_start_ungrazed_p6lzt[harv_period_z, l_idx[:,na], z_idx, 0][:,na,:]
                                           * (1-arable_l[:, na,na])
                                           * (1-np.sum(pasture_rt[:, na, :], axis=-1)))    # sum pasture proportion across the t axis to get area of crop

    ## Pasture growth, consumption of green feed.
    me_cons_grnha_fgop6lzt, volume_grnha_gop6lzt, foo_start_grnha_op6lzt, foo_end_grnha_gop6lzt, senesce_period_grnha_gop6lzt \
    , senesce_eos_grnha_gop6lzt, dmd_sward_grnha_gop6lzt, pgr_grnha_gop6lzt, foo_endprior_grnha_gop6lzt, cons_grnha_t_gop6lzt \
    , foo_ave_grnha_gop6lzt, dmd_diet_grnha_gop6lzt = pfun.f_grn_pasture(
        cu3, cu4, i_fxg_foo_op6lzt, i_fxg_pgr_op6lzt, c_pgr_gi_scalar_gp6t, grn_foo_start_ungrazed_p6lzt
        , i_foo_graze_propn_gt, grn_senesce_startfoo_p6zt, grn_senesce_pgrcons_p6zt, i_grn_senesce_eos_p6zt
        , i_base_p6t, i_grn_trampling_p6t, i_grn_dig_p6lzt, i_grn_dmd_range_p6t, i_pasture_stage_p6z
        , i_legume_zt, me_threshold_fp6zt, i_me_eff_gainlose_p6t, mask_greenfeed_exists_p6zt
        , length_fz, nv_is_not_confinement_f)
    volume_grnha_gop6lzt = volume_grnha_gop6lzt / (1 + sen.sap['pi'])


    ## dry, dmd & foo of feed consumed
    dry_mecons_t_fdp6zt, dry_volume_t_dp6zt, dry_dmd_dp6zt, dry_foo_dp6zt = pfun.f_dry_pasture(
        cu3, cu4, i_dry_dmd_ave_p6zt, i_dry_dmd_range_p6zt, i_dry_foo_high_p6zt, me_threshold_fp6zt, i_me_eff_gainlose_p6t
        , mask_dryfeed_exists_p6zt, i_pasture_stage_p6z, nv_is_not_confinement_f, i_legume_zt, n_feed_pools)
    dry_volume_t_dp6zt = dry_volume_t_dp6zt / (1 + sen.sap['pi'])

    ## dry, animal removal, mask consumption in periods where dry doesnt exist to remove the decision variable in pyomo.
    dry_removal_t_p6zt  = (1000 * (1 + i_dry_trampling_p6t[:,na,:])
                           * removal_scalar_dry_decay_p6zt
                           * mask_dryfeed_exists_p6zt)

    ## Senescence of green feed into the dry pool.
    senesce_grnha_dgop6lzt = pfun.f1_senescence(senesce_period_grnha_gop6lzt, senesce_eos_grnha_gop6lzt, dry_decay_period_p6zt
                                              , dmd_sward_grnha_gop6lzt, i_grn_dmd_senesce_redn_p6zt, dry_dmd_dp6zt
                                              , mask_greenfeed_exists_p6zt)


    ######
    #poc #
    ######
    ##call poc function - info about poc can be found in function doc string.
    poc_con_p6l, poc_md_fp6, poc_vol_p6z = pfun.f_poc(cu3, cu4, i_poc_intake_daily_p6lt, i_poc_dmd_p6t, i_poc_foo_p6t
                                                              , i_legume_zt, i_pasture_stage_p6z, nv_is_not_confinement_f)
    poc_vol_p6z = poc_vol_p6z/ (1 + sen.sap['pi'])

    ###########
    #params   #
    ###########
    ##non seasonal
    pasture_area = pasture_rt.ravel() * 1  # times 1 to convert from bool to int eg if the phase is pasture then 1ha of pasture is recorded.
    params['pasture_area_rt'] = dict(zip(index_rt,pasture_area))

    erosion_rav_p6lrt = erosion_p6lrt.ravel()
    params['p_erosion_p6lrt'] = dict(zip(index_p6lrt,erosion_rav_p6lrt))

    poc_con_rav_p6l = poc_con_p6l.ravel()
    params['p_poc_con_p6l'] = dict(zip(index_p6l,poc_con_rav_p6l))

    poc_md_rav_vf = poc_md_fp6.ravel()
    params['p_poc_md_fp6'] = dict(zip(index_fp6,poc_md_rav_vf))

    ##create season params in loop
    for z in range(len(keys_z)):
        ##create season key for params dict
        params[keys_z[z]] = {}
        scenario = keys_z[z]

        ###create param from dataframe
        params[scenario]['p_harvest_period_prop'] = harvest_period_prop[scenario].to_dict()

        ###create param from numpy

        dry_removal_t_rav_p6t = dry_removal_t_p6zt[...,z,:].ravel()
        params[scenario]['p_dry_removal_t_p6t'] = dict(zip(index_p6t,dry_removal_t_rav_p6t))

        ##convert the change in dry and green FOO at destocking and restocking into a pyomo param (for the area that is resown)
        foo_dry_reseeding_rav_dp6lrt = foo_dry_reseeding_dp6lrzt[...,z,:].ravel()
        params[scenario]['p_foo_dry_reseeding_dp6lrt'] = dict(zip(index_dp6lrt,foo_dry_reseeding_rav_dp6lrt))
        foo_grn_reseeding_rav_p6lrt = foo_grn_reseeding_p6lrzt[...,z,:].ravel()
        params[scenario]['p_foo_grn_reseeding_p6lrt'] = dict(zip(index_p6lrt,foo_grn_reseeding_rav_p6lrt))

        pas_sow_rav_p5lrk = pas_sow_p5lrkz[...,z].ravel()
        params[scenario]['p_pas_sow_p5lrk'] = dict(zip(index_plrk,pas_sow_rav_p5lrk))

        phase_area_rav_p6lrt = phase_area_p6lrzt[...,z,:].ravel()
        params[scenario]['p_phase_area_p6lrt'] = dict(zip(index_p6lrt,phase_area_rav_p6lrt))

        dry_transfer_prov_t_rav_p6t = dry_transfer_prov_t_p6zt[...,z,:].ravel()
        params[scenario]['p_dry_transfer_prov_t_p6t'] = dict(zip(index_p6t,dry_transfer_prov_t_rav_p6t))

        dry_transfer_req_t_rav_p6t = dry_transfer_req_t_p6zt[...,z,:].ravel()
        params[scenario]['p_dry_transfer_req_t_p6t'] = dict(zip(index_p6t,dry_transfer_req_t_rav_p6t))

        germination_rav_p6lrt = germination_p6lrzt[...,z,:].ravel()
        params[scenario]['p_germination_p6lrt'] = dict(zip(index_p6lrt,germination_rav_p6lrt))

        nap_rav_dp6lrt = nap_dp6lrzt[...,z,:].ravel()
        params[scenario]['p_nap_dp6lrt'] = dict(zip(index_dp6lrt,nap_rav_dp6lrt))

        foo_start_grnha_rav_op6lt = foo_start_grnha_op6lzt[...,z,:].ravel()
        params[scenario]['p_foo_start_grnha_op6lt'] = dict(zip(index_op6lt ,foo_start_grnha_rav_op6lt))

        foo_end_grnha_rav_gop6lt = foo_end_grnha_gop6lzt[...,z,:].ravel()
        params[scenario]['p_foo_end_grnha_gop6lt'] = dict( zip(index_gop6lt ,foo_end_grnha_rav_gop6lt))

        me_cons_grnha_rav_fgop6lt = me_cons_grnha_fgop6lzt[...,z,:].ravel()
        params[scenario]['p_me_cons_grnha_fgop6lt'] = dict(zip(index_fgop6lt,me_cons_grnha_rav_fgop6lt))

        dry_mecons_t_rav_fdp6t = dry_mecons_t_fdp6zt[...,z,:].ravel()
        params[scenario]['p_dry_mecons_t_fdp6t'] = dict(zip(index_fdp6t,dry_mecons_t_rav_fdp6t))

        volume_grnha_rav_gop6lt = volume_grnha_gop6lzt[...,z,:].ravel()
        params[scenario]['p_volume_grnha_gop6lt'] = dict(zip(index_gop6lt,volume_grnha_rav_gop6lt))

        dry_volume_t_rav_dp6t = dry_volume_t_dp6zt[...,z,:].ravel()
        params[scenario]['p_dry_volume_t_dp6t'] = dict(zip(index_dp6t,dry_volume_t_rav_dp6t))

        senesce_grnha_rav_dgop6lt = senesce_grnha_dgop6lzt[...,z,:].ravel()
        params[scenario]['p_senesce_grnha_dgop6lt'] = dict(zip(index_dgop6lt,senesce_grnha_rav_dgop6lt))

        poc_vol_rav_p6 = poc_vol_p6z[...,z].ravel()
        params[scenario]['p_poc_vol_p6'] = dict(zip(keys_p6,poc_vol_rav_p6))

    ###########
    #report   #
    ###########
    ##keys
    r_vals['keys_d'] = keys_d
    r_vals['keys_f'] = keys_f
    r_vals['keys_p6'] = keys_p6
    r_vals['keys_g'] = keys_g
    r_vals['keys_l'] = keys_l
    r_vals['keys_o'] = keys_o
    r_vals['keys_p5'] = keys_p5
    r_vals['keys_r'] = keys_r
    r_vals['keys_t'] = keys_t
    r_vals['keys_k'] = keys_k

    ##store report vals
    r_vals['pasture_area_rt'] = pasture_rt
    r_vals['keys_pastures'] = pastures
    r_vals['days_p6z'] = length_fz

    r_vals['pgr_grnha_gop6lzt'] = pgr_grnha_gop6lzt
    r_vals['foo_end_grnha_gop6lzt'] = foo_endprior_grnha_gop6lzt #Green FOO prior to eos senescence
    r_vals['cons_grnha_t_gop6lzt'] = cons_grnha_t_gop6lzt
    r_vals['nv_grnha_fgop6lzt'] = fun.f_divide(me_cons_grnha_fgop6lzt, volume_grnha_gop6lzt)
    r_vals['nv_dry_fdp6zt'] = fun.f_divide(dry_mecons_t_fdp6zt, dry_volume_t_dp6zt)
    r_vals['foo_ave_grnha_gop6lzt'] = foo_ave_grnha_gop6lzt
    r_vals['dmd_diet_grnha_gop6lzt'] = dmd_diet_grnha_gop6lzt
    r_vals['dry_foo_dp6zt'] = dry_foo_dp6zt
    r_vals['dry_dmd_dp6zt'] = dry_dmd_dp6zt