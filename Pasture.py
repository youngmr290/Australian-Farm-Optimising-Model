"""
@author: young

Description of this pasture module: This representation includes at optimisation (ie the following options are represented in the variables of the model)
    Growth rate of pasture (PGR) varies with FOO at the start of the period and grazing intensity during the period
        Grazing intensity operates by altering the average FOO during the period
    The nutritive value of the green feed consumed (as represented by ME & volume) varies with FOO & grazing intensity.
        Grazing intensity alters the average FOO during the period and the capacity of the animals to select a higher quality diet.
    Selective grazing of dry pasture. 2 dry pasture quality pools are represented and either can be selected for grazing
        Note: There is not a constraint that ensures that the high quality pool is grazed prior to the low quality pool (as there is in the stubble selective grazing)

Pasture is a primary livestock feed source because in an extensive system it is a good source of megajoules per doll ar. In AFO the biological
details of pasture are represented in detail. Including the relationship between FOO, PGR and quality, the effects of
rotation on pasture production, the life cycle over the year, pasture conservation and pasture availability to livestock.
Pasture is split into the following sections:

    #. Germination & Reseeding of pasture
    #. Green & dry feed: growth, senescence & consumption
    #. Pasture consumed on crop paddocks
    #. Limit on grazing for soil conservation

The green pasture activity represents FOO at the start of the period, FOO at the end of the period, animal removal,
energy per unit of dry matter and volume. These aspects are aggregated to allow the effects of FOO level and level
of defoliation on diet quality to be included. The reasons for this are:

    #. The intake capacity of livestock is affected by the level of FOO (e.g. when there is more FOO, animals can eat
       more and achieve higher growth rates).
    #. Livestock diet quality change with grazing pressure (eg. by running a lower stocking rate livestock can improve
       their diet quality through increased diet selectivity). This selectivity can be important for finishing animals
       for market or fattening animals for mating.
    #. The digestibility of pasture decreases as the length of time from the last defoliation increases (i.e. older
       leaves are less digestible). Having consumption and FOO in the same activity allows a drop in digestibility
       associated with old leaf to be included by linking digestibility to FOO. This is especially important for species
       such as kikuyu that drop in digestibility rapidly if pastures are grazed laxly and FOO increases.

These issues are more important in a system producing meat, where growth rate of animals and hence diet quality is
critical to profitability. In a meat system the trade-off between quantity of feed utilised and quality of feed is
quite different than the trade-off for a wool system.
For a given period the activities are defined by starting FOO level and grazing intensity. There are three foo levels;
low, medium and high starting FOO and four grazing intensities; no grazing, low, medium and high. Green pasture
activities represent the total green pasture on the farm in each period. The activities are initially provided based
on the area of pasture and its level of establishment. Pasture establishment is dependant on the rotation history and
whether or not the pasture is resown. Additionally, the option to manipulate pasture (e.g. spraytop) is included. Each
different management option (i.e. reseeding and manipulation) are represented as individual land use options so the
model can optimise which to select. When the pasture senesces, it is removed from the green activities and allocated
into the dry pasture activities. Dry pasture is represented by a low and high quality activities. Over time the quality
of the dry pasture changes and the quantity changes as it decays and is consumed.
AFO can handle multiple pasture types. The user simply needs to create a copy of the inputs and calibrate them to the
new pasture.


"""

#todo add labour required for feed budgeting. Inputs are currently in the sheep sheet of Property.xls


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

#todo Will need to add the foo reduction in the current year for manipulated pasture and a germination reduction in the following year.

def f_pasture(params, r_vals, ev):
    ######################
    ##background vars    #
    ######################
    na = np.newaxis

    ########################
    ##ev stuff             #
    ########################
    len_ev = ev['len_ev']
    ev_is_not_confinement_v = np.full(len_ev, True)
    ev_is_not_confinement_v[-1] = np.logical_not(ev['confinement_inc']) #if confinment period is included the last fev pool is confinment.

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
    n_feed_pools    = len_ev
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
    # length_f  = np.array(pinp.period['feed_periods'].loc[:pinp.period['feed_periods'].index[-2],'length']) # not including last row because that is the start of the following year. #todo as above this will need z axis
    # feed_period_dates_f = np.array(i_feed_period_dates,dtype='datetime64[D]')
    length_fz  = np.array(per.f_feed_periods(option=1),dtype='float64')
    feed_period_dates_fz = fun.f_baseyr(per.f_feed_periods()).astype('datetime64[D]') #feed periods are all date to the base yr (eg 2019) - this is required for some of the allocation formulas

    # vgoflt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dgoflzt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # vdft    = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
    vfzt    = (n_feed_pools, n_feed_periods, n_season_types, n_pasture_types)
    # dft     = (n_dry_groups, n_feed_periods, n_pasture_types)
    # goflzt  = (n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # goft    = (n_grazing_int, n_foo_levels, n_feed_periods, n_pasture_types)
    gft     = (n_grazing_int, n_feed_periods, n_pasture_types)
    gt      = (n_grazing_int, n_pasture_types)
    oflzt   = (n_foo_levels, n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    # dflrt   = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    dflrzt  = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    flrt    = (n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    flrzt   = (n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    # frt     = (n_feed_periods, n_phases_rotn, n_pasture_types)
    rt      = (n_phases_rotn, n_pasture_types)
    flt     = (n_feed_periods, n_lmu, n_pasture_types)
    flzt    = (n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    lt      = (n_lmu, n_pasture_types)
    lzt     = (n_lmu, n_season_types, n_pasture_types)
    ft      = (n_feed_periods, n_pasture_types)
    fzt     = (n_feed_periods, n_season_types, n_pasture_types)
    zt      = (n_season_types, n_pasture_types)
    # t       = (n_pasture_types)

    ## define the vessels that will store the input data that require pre-defining
    ### all need pre-defining because inputs are in separate pasture type arrays
    i_phase_germ_dict = dict()
    i_grn_senesce_daily_ft      = np.zeros(ft,  dtype = 'float64')              # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_fzt       = np.zeros(fzt,  dtype = 'float64')             # proportion of green feed that senesces in period (due to a water deficit or completing life cycle)
    dry_decay_daily_fzt         = np.zeros(fzt,  dtype = 'float64')             # daily decline in dry foo in each period
    i_end_of_gs_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_exists_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_decay_t               = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    # i_me_maintenance_vft        = np.zeros(vft,  dtype = 'float64')     # M/D level for target LW pattern
    c_pgr_gi_scalar_gft         = np.zeros(gft,  dtype = 'float64')     # pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_graze_propn_gt        = np.zeros(gt, dtype ='float64')        # proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_oflzt             = np.zeros(oflzt, dtype = 'float64')    # FOO level     for the FOO/growth/grazing variables.
    i_fxg_pgr_oflzt             = np.zeros(oflzt, dtype = 'float64')    # PGR level     for the FOO/growth/grazing variables.
    c_fxg_a_oflzt               = np.zeros(oflzt, dtype = 'float64')    # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_oflzt               = np.zeros(oflzt, dtype = 'float64')    # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_oflt               = np.zeros(oflt, dtype = 'float64')     # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_oflt               = np.zeros(oflt, dtype = 'float64')     # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_flzt              = np.zeros(flzt, dtype = 'float64') # green pasture digestibility in each period, LMU, season & pasture type.
    i_poc_intake_daily_flt      = np.zeros(flt, dtype = 'float64')  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_flt      = np.zeros(flt, dtype = 'float64')  # minimum foo at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lzt           = np.zeros(lzt,  dtype = 'float64') # scale the germination levels for each lmu
    i_restock_fooscalar_lt      = np.zeros(lt,  dtype = 'float64')  # scalar for FOO between LMUs when pastures are restocked after reseeding

    i_me_eff_gainlose_ft        = np.zeros(ft,  dtype = 'float64')  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_ft          = np.zeros(ft,  dtype = 'float64')  # green pasture trampling in each feed period as proportion of intake.
    i_dry_trampling_ft          = np.zeros(ft,  dtype = 'float64')  # dry pasture trampling   in each feed period as proportion of intake.
    i_base_ft                   = np.zeros(ft,  dtype = 'float64')  # lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_ft     = np.zeros(ft,  dtype = 'float64')  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_ft          = np.zeros(ft,  dtype = 'float64')  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_fzt  = np.zeros(fzt,  dtype = 'float64') # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_fzt           = np.zeros(fzt,  dtype = 'float64') # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_fzt         = np.zeros(fzt,  dtype = 'float64') # range in digestibility of dry feed if it is not grazed
    i_dry_foo_high_fzt          = np.zeros(fzt,  dtype = 'float64') # expected foo for the dry pasture in the high quality pool
    dry_decay_period_fzt        = np.zeros(fzt,  dtype = 'float64') # decline in dry foo for each period
    mask_dryfeed_exists_fzt     = np.zeros(fzt,  dtype = bool)      # mask for period when dry feed exists
    mask_greenfeed_exists_fzt   = np.zeros(fzt,  dtype = bool)      # mask for period when green feed exists
    i_germ_scalar_fzt           = np.zeros(fzt,  dtype = 'float64') # allocate the total germination between feed periods
    i_grn_cp_ft                 = np.zeros(ft,  dtype = 'float64')  # crude protein content of green feed
    i_dry_cp_ft                 = np.zeros(ft,  dtype = 'float64')  # crude protein content of dry feed
    i_poc_dmd_ft                = np.zeros(ft,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_ft                = np.zeros(ft,  dtype = 'float64')  # foo of pasture consumed on crop paddocks
    # grn_senesce_startfoo_ft     = np.zeros(ft,  dtype = 'float64')  # proportion of the FOO at the start of the period that senesces during the period
    # grn_senesce_pgrcons_ft      = np.zeros(ft,  dtype = 'float64')  # proportion of the (total or average daily) PGR that senesces during the period (consumption leads to a reduction in senescence)

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
    i_fec_maintenance_t         = np.zeros(n_pasture_types, dtype = 'float64')  # approximate feed energy concentration for maintenance (FEC = M/D * relative intake)

#    germination_flrzt           = np.zeros(flrzt,  dtype = 'float64')  # germination for each rotation phase (kg/ha)
    foo_grn_reseeding_flrzt     = np.zeros(flrzt,  dtype = 'float64')  # green FOO adjustment for destocking and restocking of the resown area (kg/ha)
    foo_dry_reseeding_flrzt     = np.zeros(flrzt,  dtype = 'float64')  # dry FOO adjustment for destocking and restocking of the resown area (kg/ha)
    # foo_dry_reseeding_dflrzt    = np.zeros(dflrzt, dtype = 'float64')  # dry FOO adjustment allocated to the low & high quality dry feed pools (kg/ha)
    # dry_removal_t_ft            = np.zeros(ft,   dtype = 'float64')  # Total DM removal from the tonne consumed (includes trampling)

    ### define the array that links rotation phase and pasture type
    pasture_rt                  = np.zeros(rt, dtype = 'float64')


    ## create numpy index for param dicts ^creating indexes is a bit slow
    ### the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
    keys_d  = np.asarray(sinp.general['dry_groups'])
    keys_v  = np.array(['fev{0}' .format(i) for i in range(len_ev)])
    keys_f  = pinp.period['i_fp_idx']
    keys_g  = np.asarray(sinp.general['grazing_int'])
    keys_l  = np.array(pinp.general['lmu_area'].index[lmu_mask_l]).astype('str')    # lmu index description
    keys_o  = np.asarray(sinp.general['foo_levels'])
    keys_p  = np.array(per.p_date2_df().index).astype('str')
    keys_r  = np.array(phases_rotn_df.index).astype('str')
    keys_t  = np.asarray(pastures)                                      # pasture type index description
    keys_k  = np.asarray(list(sinp.landuse['All']))                     #landuse
    keys_z  = pinp.f_keys_z()

    ### plrk
    arrays=[keys_p, keys_l, keys_r, keys_k]
    index_plrk=fun.cartesian_product_simple_transpose(arrays)
    index_plrk=tuple(map(tuple, index_plrk)) #create a tuple rather than a list because tuples are faster

    ### rt
    arrays=[keys_r, keys_t]
    index_rt=fun.cartesian_product_simple_transpose(arrays)
    index_rt=tuple(map(tuple, index_rt)) #create a tuple rather than a list because tuples are faster

    ### flrt
    arrays=[keys_f, keys_l, keys_r, keys_t]
    index_flrt=fun.cartesian_product_simple_transpose(arrays)
    index_flrt=tuple(map(tuple, index_flrt)) #create a tuple rather than a list because tuples are faster

    ### oflt
    arrays=[keys_o, keys_f, keys_l, keys_t]
    index_oflt=fun.cartesian_product_simple_transpose(arrays)
    index_oflt=tuple(map(tuple, index_oflt)) #create a tuple rather than a list because tuples are faster

    ### goflt
    arrays=[keys_g, keys_o, keys_f, keys_l, keys_t]
    index_goflt=fun.cartesian_product_simple_transpose(arrays)
    index_goflt=tuple(map(tuple, index_goflt)) #create a tuple rather than a list because tuples are faster

    ### vgoflt
    arrays=[keys_v, keys_g, keys_o, keys_f, keys_l, keys_t]
    index_vgoflt=fun.cartesian_product_simple_transpose(arrays)
    index_vgoflt=tuple(map(tuple, index_vgoflt)) #create a tuple rather than a list because tuples are faster

    ### dgoflt
    arrays=[keys_d, keys_g, keys_o, keys_f, keys_l, keys_t]
    index_dgoflt=fun.cartesian_product_simple_transpose(arrays)
    index_dgoflt=tuple(map(tuple, index_dgoflt)) #create a tuple rather than a list because tuples are faster

    ### dflrt
    arrays=[keys_d, keys_f, keys_l, keys_r, keys_t]
    index_dflrt=fun.cartesian_product_simple_transpose(arrays)
    index_dflrt=tuple(map(tuple, index_dflrt)) #create a tuple rather than a list because tuples are faster

    ### vdft
    arrays=[keys_v, keys_d, keys_f, keys_t]
    index_vdft=fun.cartesian_product_simple_transpose(arrays)
    index_vdft=tuple(map(tuple, index_vdft)) #create a tuple rather than a list because tuples are faster

    ### dft
    arrays=[keys_d, keys_f, keys_t]
    index_dft=fun.cartesian_product_simple_transpose(arrays)
    index_dft=tuple(map(tuple, index_dft)) #create a tuple rather than a list because tuples are faster

    ### vf
    arrays=[keys_v, keys_f]
    index_vf=fun.cartesian_product_simple_transpose(arrays)
    index_vf=tuple(map(tuple, index_vf)) #create a tuple rather than a list because tuples are faster

    ### fl
    arrays=[keys_f, keys_l]
    index_fl=fun.cartesian_product_simple_transpose(arrays)
    index_fl=tuple(map(tuple, index_fl)) #create a tuple rather than a list because tuples are faster

    ### ft
    arrays=[keys_f, keys_t]
    index_ft=fun.cartesian_product_simple_transpose(arrays)
    index_ft=tuple(map(tuple, index_ft)) #create a tuple rather than a list because tuples are faster

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
        i_poc_intake_daily_flt[...,t]       = exceldata['POCCons'][:,lmu_mask_l]
        i_legume_zt[...,t]                  = pinp.f_seasonal_inp(exceldata['Legume'], numpy=True)
        i_restock_grn_propn_t[t]            = exceldata['FaG_PropnGrn']
        i_grn_dmd_senesce_redn_fzt[...,t]   = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigRednSenesce'],0,1), numpy=True, axis=1)
        i_dry_dmd_ave_fzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryAve'],0,1), numpy=True, axis=1)
        i_dry_dmd_range_fzt[...,t]          = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryRange'],0,1), numpy=True, axis=1)
        i_dry_foo_high_fzt[...,t]           = pinp.f_seasonal_inp(np.swapaxes(exceldata['FOODryH'],0,1), numpy=True, axis=1)
        i_germ_scalar_fzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarFP'],0,1), numpy=True, axis=1)

        i_grn_cp_ft[...,t]                  = exceldata['CPGrn']
        i_dry_cp_ft[...,t]                  = exceldata['CPDry']
        i_poc_dmd_ft[...,t]                 = exceldata['DigPOC']
        i_poc_foo_ft[...,t]                 = exceldata['FOOPOC']
        i_germ_scalar_lzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarLMU'],0,1), numpy=True, axis=1)[lmu_mask_l,...]
        i_restock_fooscalar_lt[...,t]       = exceldata['FaG_LMU'][lmu_mask_l]  #todo may need a z axis

        i_lmu_conservation_flt[...,t]       = exceldata['ErosionLimit'][:, lmu_mask_l]

        i_reseeding_date_start_zt[...,t]    = pinp.f_seasonal_inp(exceldata['Date_Seeding'], numpy=True)
        i_reseeding_date_end_zt[...,t]      = pinp.f_seasonal_inp(exceldata['pas_seeding_end'], numpy=True)
        i_destock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_Destocking'], numpy=True)
        i_destock_foo_zt[...,t]             = pinp.f_seasonal_inp(exceldata['FOOatSeeding'], numpy=True) #ungrazed foo when destocked for reseeding
        i_restock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_ResownGrazing'], numpy=True)
        i_restock_foo_arable_t[t]           = exceldata['FOOatGrazing']

        i_grn_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_ft[...,t]       = np.asfarray(exceldata['SenescePropn'])
        i_grn_senesce_eos_fzt[...,t]        = pinp.f_seasonal_inp(np.asfarray(exceldata['SenesceEOS']), numpy=True, axis=1)
        i_base_ft[...,t]                    = np.asfarray(exceldata['BaseLevelInput'])
        i_grn_dmd_declinefoo_ft[...,t]      = np.asfarray(exceldata['DigDeclineFOO'])
        i_grn_dmd_range_ft[...,t]           = np.asfarray(exceldata['DigSpread'])
        i_foo_graze_propn_gt[..., t]        = np.asfarray(exceldata['FOOGrazePropn'])
        #### impact of grazing intensity (at the other levels) on PGR during the period
        c_pgr_gi_scalar_gft[...,t]      = 1 - i_foo_graze_propn_gt[..., na, t] ** 2 * (1 - np.asfarray(exceldata['PGRScalarH']))

        i_fxg_foo_oflzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowFOO'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_foo_oflzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedFOO'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_me_eff_gainlose_ft[...,t]     = exceldata['MaintenanceEff'][:,0]
        # i_me_maintenance_vft[...,t]     = exceldata['MaintenanceEff'].iloc[:,1:].to_numpy().T  # replaced by the ev_cutoff. Still used in PastureTest
        i_fec_maintenance_t[t]          = exceldata['MaintenanceFEC']
        ## # i_fxg_foo_oflt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_oflzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_pgr_oflzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]
        i_fxg_pgr_oflzt[2,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_flzt[...,t]           = pinp.f_seasonal_inp(np.moveaxis(exceldata['DigGrn'],0,-1), numpy=True, axis=-1)[:,lmu_mask_l,...]  # numpy array of inputs for green pasture digestibility on each LMU.

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
    mask_dryfeed_exists_fzt[...] = index_f[:, na, na] >= i_dry_exists_zt   #mask periods when dry feed is available to livestock.
    mask_greenfeed_exists_fzt[...] = index_f[:, na, na] <= i_end_of_gs_zt   #green exists in the period which is the end of growing season hence <=

    ### calculate dry_decay_period (used in reseeding and green&dry)
    ### dry_decay_daily is decay of dry foo at the start of the period that was transferred in from senescence in the previous period.
    ### dry_decay_daily does not effect green feed that sceneses during the current period.
    dry_decay_daily_fzt[...] = i_dry_decay_t
    for t in range(n_pasture_types):
        for z in range(n_season_types):
            dry_decay_daily_fzt[0:i_dry_exists_zt[z,t], z, t] = 1  #couldn't do this without loops - advanced indexing doesnt appear to work when taking multiple slices
    dry_decay_period_fzt[...] = 1 - (1 - dry_decay_daily_fzt) ** length_fz[...,na]
    ## dry, DM decline (high = low pools)
    #todo look at masking the dry transfer to only those periods that dry exist (decay eos > 0)
    dry_transfer_t_fzt = 1000 * (1-dry_decay_period_fzt)

    ###create equation coefficients for pgr = a+b*foo
    i_fxg_foo_oflzt[2,...]  = 100000 #large number so that the np.searchsorted doesn't go above
    c_fxg_b_oflzt[0,...] =  fun.f_divide(i_fxg_pgr_oflzt[0,...], i_fxg_foo_oflzt[0,...])
    c_fxg_b_oflzt[1,...] =   fun.f_divide((i_fxg_pgr_oflzt[1,...] - i_fxg_pgr_oflzt[0,...])
                            , (i_fxg_foo_oflzt[1,...] - i_fxg_foo_oflzt[0,...]))
    c_fxg_b_oflzt[2,...] =  0
    c_fxg_a_oflzt[0,...] =  0
    c_fxg_a_oflzt[1,...] =  i_fxg_pgr_oflzt[0,...] - c_fxg_b_oflzt[1,...] * i_fxg_foo_oflzt[0,...]
    c_fxg_a_oflzt[2,...] =  i_fxg_pgr_oflzt[1,...] # because slope = 0

    ## proportion of start foo that senesces during the period, different formula than excel
    grn_senesce_startfoo_fzt = 1 - ((1 - i_grn_senesce_daily_ft[:,na,:]) **  length_fz[...,na])

    ## average senescence over the period for the growth and consumption
    grn_senesce_pgrcons_fzt = 1 - ((1 -(1 - i_grn_senesce_daily_ft[:,na,:]) ** (length_fz[...,na]+1))
                                   /        i_grn_senesce_daily_ft[:,na,:]-1) / length_fz[...,na]



    #################################################
    #Calculate germination and reseeding parameters #
    #################################################


    ## define instantiate arrays that are assigned in slices
    na_erosion_flrt      = np.zeros(flrt,  dtype = 'float64')
    # na_phase_area_flrzt  = np.zeros(flrzt, dtype = 'float64')
    # grn_restock_foo_flzt = np.zeros(flzt,  dtype = 'float64')
    # dry_restock_foo_flzt = np.zeros(flzt,  dtype = 'float64')
    foo_na_destock_fzt   = np.zeros(fzt,   dtype = 'float64')
    germ_scalar_rt       = np.zeros(rt,    dtype='float64')
    resown_rt            = np.zeros(rt,    dtype='int')

    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df

    germination_flrzt, max_germination_flz = pfun.f_germination(i_germination_std_zt, i_germ_scalar_lzt, germ_scalar_rt
                                                                , i_germ_scalar_fzt, pasture_rt, arable_l,  resown_rt
                                                                , pastures, phase_germresow_df, i_phase_germ_dict)

    foo_grn_reseeding_flrzt, foo_dry_reseeding_dflrzt, periods_destocked_fzt = pfun.f_reseeding(
        i_destock_date_zt, i_restock_date_zt, i_destock_foo_zt, i_restock_grn_propn_t, resown_rt, feed_period_dates_fz
        , foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt, foo_na_destock_fzt, i_restock_fooscalar_lt
        , i_restock_foo_arable_t, dry_decay_period_fzt, i_fxg_foo_oflzt, c_fxg_a_oflzt, c_fxg_b_oflzt, i_grn_senesce_eos_fzt
        , grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt, length_fz, n_feed_periods, max_germination_flz
        , t_idx, z_idx, l_idx)

    ## sow param determination
    pas_sow_plrkz = pfun.f_pas_sow(i_reseeding_date_start_zt, i_reseeding_date_end_zt, resown_rt, arable_l, phases_rotn_df)

    ## area of green pasture being grazed and growing
    phase_area_flrzt = pfun.f_green_area(resown_rt, pasture_rt, periods_destocked_fzt, arable_l)

    ## erosion limit. The minimum FOO at the end of each period#
    erosion_flrt = pfun.f_erosion(i_lmu_conservation_flt, arable_l, pasture_rt)

    ## initialise numpy arrays used only in this method
    senesce_propn_dgoflzt      = np.zeros(dgoflzt, dtype = 'float64')
    nap_dflrzt                 = np.zeros(dflrzt,  dtype = 'float64')
    me_threshold_vfzt          = np.zeros(vfzt,    dtype = 'float64')   # the threshold for the EV pools which define the animals feed quality requirements

    ## create numpy array of threshold values from the ev dictionary
    ### note: v in pasture is f in StockGen and f in pasture is p6 in StockGen
    me_threshold_vfzt[...] = np.swapaxes(ev['fev_cutoff_ave_p6f'], axis1=0, axis2=1)[...,na,na]
    ### if the threshold is below the expected maintenance quality set to the maintenance quality
    ### switching from one below maintenance feed to another that is further below maintenance doesn't affect average efficiency
    me_threshold_vfzt[me_threshold_vfzt < i_fec_maintenance_t] = i_fec_maintenance_t

    ## FOO on the non-arable areas in crop paddocks is ungrazed FOO of pasture type 0 (annual), therefore calculate the profile based on the pasture type 0 values
    grn_foo_start_ungrazed_flzt, dry_foo_start_ungrazed_flzt = pfun.f1_calc_foo_profile(
        max_germination_flz[..., na], dry_decay_period_fzt[..., 0:1], length_fz, i_fxg_foo_oflzt[..., 0:1]
        , c_fxg_a_oflzt[..., 0:1], c_fxg_b_oflzt[..., 0:1], i_grn_senesce_eos_fzt[..., 0:1], grn_senesce_startfoo_fzt[..., 0:1]
        , grn_senesce_pgrcons_fzt[..., 0:1])

    ### non arable pasture becomes available to graze at the beginning of the first harvest period
    # harvest_period  = fun.period_allocation(pinp.period['feed_periods']['date'], range(len(pinp.period['feed_periods'])), pinp.period['harv_date']) #use range(len()) to get the row number that harvest occurs has to be row number not index name because it is used to index numpy below
    harv_period_z, harv_proportion_z = fun.period_proportion_np(feed_period_dates_fz, harv_date_z)
    index = pd.MultiIndex.from_arrays([keys_f[harv_period_z], keys_z])
    harvest_period_prop = pd.Series(harv_proportion_z, index=index).unstack()
    # params['p_harvest_period_prop']  = dict([(pinp.period['feed_periods'].index[harv_period_z], harv_proportion_z)])

    ### all pasture from na area goes into the Low pool (#1) because it is rank & low quality
    nap_dflrzt[0, harv_period_z, l_idx[:,na,na], r_idx[:,na], z_idx, 0] = (
                                             dry_foo_start_ungrazed_flzt[harv_period_z, l_idx[:,na], z_idx, 0][:,na,:]
                                           * (1-arable_l[:, na,na])
                                           * (1-np.sum(pasture_rt[:, na, :], axis=-1)))    # sum pasture proportion across the t axis to get area of crop

    ## Pasture growth, consumption of green feed.
    me_cons_grnha_vgoflzt, volume_grnha_goflzt, senesce_period_grnha_goflzt, senesce_eos_grnha_goflzt, dmd_sward_grnha_goflzt     \
         = pfun.f_grn_pasture(cu3, cu4, i_fxg_foo_oflzt, i_fxg_pgr_oflzt, c_pgr_gi_scalar_gft, grn_foo_start_ungrazed_flzt
                              , i_foo_graze_propn_gt, grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt, i_grn_senesce_eos_fzt
                              , i_base_ft, i_grn_trampling_ft, i_grn_dig_flzt, i_grn_dmd_range_ft, i_pasture_stage_p6z
                              , i_legume_zt, me_threshold_vfzt, i_me_eff_gainlose_ft, mask_greenfeed_exists_fzt
                              , length_fz, ev_is_not_confinement_v)


    ## dry, dmd & foo of feed consumed
    dry_mecons_t_vdfzt, dry_volume_t_dfzt, dry_dmd_dfzt = pfun.f_dry_pasture(cu3, cu4, i_dry_dmd_ave_fzt, i_dry_dmd_range_fzt
                                , i_dry_foo_high_fzt, me_threshold_vfzt, i_me_eff_gainlose_ft, mask_dryfeed_exists_fzt
                                , i_pasture_stage_p6z, ev_is_not_confinement_v, i_legume_zt, n_feed_pools)

    ## dry, animal removal
    dry_removal_t_ft  = 1000 * (1 + i_dry_trampling_ft)

    ## Senescence of green feed into the dry pool.
    senesce_grnha_dgoflzt = pfun.f_senescence(senesce_period_grnha_goflzt, senesce_eos_grnha_goflzt, dry_decay_period_fzt
                                              , dmd_sward_grnha_goflzt, i_grn_dmd_senesce_redn_fzt, dry_dmd_dfzt
                                              , mask_greenfeed_exists_fzt)


    ######
    #poc #
    ######
    ##call poc function - info about poc can be found in function doc string.
    poc_con_fl, poc_md_vf, poc_vol_fz = pfun.f_poc(cu3, cu4, i_poc_intake_daily_flt, i_poc_dmd_ft, i_poc_foo_ft
                                                              , i_legume_zt, i_pasture_stage_p6z, ev_is_not_confinement_v)

    ###########
    #params   #
    ###########
    ##non seasonal
    pasture_area = pasture_rt.ravel() * 1  # times 1 to convert from bool to int eg if the phase is pasture then 1ha of pasture is recorded.
    params['pasture_area_rt'] = dict(zip(index_rt,pasture_area))

    erosion_rav_flrt = erosion_flrt.ravel()
    params['p_erosion_flrt'] = dict(zip(index_flrt,erosion_rav_flrt))

    dry_removal_t_rav_ft = dry_removal_t_ft.ravel()
    params['p_dry_removal_t_ft'] = dict(zip(index_ft,dry_removal_t_rav_ft))

    poc_con_rav_fl = poc_con_fl.ravel()
    params['p_poc_con_fl'] = dict(zip(index_fl,poc_con_rav_fl))

    poc_md_rav_vf = poc_md_vf.ravel()
    params['p_poc_md_vf'] = dict(zip(index_vf,poc_md_rav_vf))

    ##create season params in loop
    for z in range(len(keys_z)):
        ##create season key for params dict
        params[keys_z[z]] = {}
        scenario = keys_z[z]

        ###create param from dataframe
        params[scenario]['p_harvest_period_prop'] = harvest_period_prop[scenario].to_dict()

        ###create param from numpy

        ##convert the change in dry and green FOO at destocking and restocking into a pyomo param (for the area that is resown)
        foo_dry_reseeding_rav_dflrt = foo_dry_reseeding_dflrzt[...,z,:].ravel()
        params[scenario]['p_foo_dry_reseeding_dflrt'] = dict(zip(index_dflrt,foo_dry_reseeding_rav_dflrt))
        foo_grn_reseeding_rav_flrt = foo_grn_reseeding_flrzt[...,z,:].ravel()
        params[scenario]['p_foo_grn_reseeding_flrt'] = dict(zip(index_flrt,foo_grn_reseeding_rav_flrt))

        pas_sow_rav_plrk = pas_sow_plrkz[...,z].ravel()
        params[scenario]['p_pas_sow_plrk'] = dict(zip(index_plrk,pas_sow_rav_plrk))

        phase_area_rav_flrt = phase_area_flrzt[...,z,:].ravel()
        params[scenario]['p_phase_area_flrt'] = dict(zip(index_flrt,phase_area_rav_flrt))

        dry_transfer_t_rav_ft = dry_transfer_t_fzt[...,z,:].ravel()
        params[scenario]['p_dry_transfer_t_ft'] = dict(zip(index_ft,dry_transfer_t_rav_ft))

        germination_rav_flrt = germination_flrzt[...,z,:].ravel()
        params[scenario]['p_germination_flrt'] = dict(zip(index_flrt,germination_rav_flrt))

        nap_rav_dflrt = nap_dflrzt[...,z,:].ravel()
        params[scenario]['p_nap_dflrt'] = dict(zip(index_dflrt,nap_rav_dflrt))

        foo_start_grnha_rav_oflt = foo_start_grnha_oflzt[...,z,:].ravel()
        params[scenario]['p_foo_start_grnha_oflt'] = dict(zip(index_oflt ,foo_start_grnha_rav_oflt))

        foo_end_grnha_rav_goflt = foo_end_grnha_goflzt[...,z,:].ravel()
        params[scenario]['p_foo_end_grnha_goflt'] = dict( zip(index_goflt ,foo_end_grnha_rav_goflt))

        me_cons_grnha_rav_vgoflt = me_cons_grnha_vgoflzt[...,z,:].ravel()
        params[scenario]['p_me_cons_grnha_vgoflt'] = dict(zip(index_vgoflt,me_cons_grnha_rav_vgoflt))

        dry_mecons_t_rav_vdft = dry_mecons_t_vdfzt[...,z,:].ravel()
        params[scenario]['p_dry_mecons_t_vdft'] = dict(zip(index_vdft,dry_mecons_t_rav_vdft))

        volume_grnha_rav_goflt = volume_grnha_goflzt[...,z,:].ravel()
        params[scenario]['p_volume_grnha_goflt'] = dict(zip(index_goflt,volume_grnha_rav_goflt))

        dry_volume_t_rav_dft = dry_volume_t_dfzt[...,z,:].ravel()
        params[scenario]['p_dry_volume_t_dft'] = dict(zip(index_dft,dry_volume_t_rav_dft))

        senesce_grnha_rav_dgoflt = senesce_grnha_dgoflzt[...,z,:].ravel()
        params[scenario]['p_senesce_grnha_dgoflt'] = dict(zip(index_dgoflt,senesce_grnha_rav_dgoflt))

        poc_vol_rav_f = poc_vol_fz[...,z].ravel()
        params[scenario]['p_poc_vol_f'] = dict(zip(keys_f,poc_vol_rav_f))

    ###########
    #report   #
    ###########
    ##keys
    r_vals['keys_d'] = keys_d
    r_vals['keys_v'] = keys_v
    r_vals['keys_f'] = keys_f
    r_vals['keys_g'] = keys_g
    r_vals['keys_l'] = keys_l
    r_vals['keys_o'] = keys_o
    r_vals['keys_p'] = keys_p
    r_vals['keys_r'] = keys_r
    r_vals['keys_t'] = keys_t
    r_vals['keys_k'] = keys_k

    ##store report vals
    r_vals['pasture_area_rt'] = pasture_rt
    r_vals['keys_pastures'] = pastures
    r_vals['days_p6z'] = length_fz

    r_vals['pgr_grnha_goflzt'] = pgr_grnha_goflzt
    r_vals['foo_end_grnha_goflzt'] = foo_endprior_grnha_goflzt #Green FOO prior to eos senescence
    r_vals['cons_grnha_t_goflzt'] = cons_grnha_t_goflzt
    r_vals['fec_grnha_vgoflzt'] = fun.f_divide(me_cons_grnha_vgoflzt, volume_grnha_goflzt)
    r_vals['fec_dry_vdfzt'] = fun.f_divide(dry_mecons_t_vdfzt, dry_volume_t_dfzt)
    r_vals['foo_ave_grnha_goflzt'] = foo_ave_grnha_goflzt
    r_vals['dmd_diet_grnha_goflzt'] = dmd_diet_grnha_goflzt
    r_vals['dry_foo_dfzt'] = dry_foo_dfzt
    r_vals['dry_dmd_dfzt'] = dry_dmd_dfzt