# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:19:20 2019

module: create the model. this then gets used in all sheets that use pyomo

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete in case we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem


@author: young
"""


#python modules
from pyomo.environ import *
import pandas as pd
import numpy as np

#AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp
import Periods as per


'''
pyomo sets
'''
##define sets - sets are redefined for each exp in case they change due to SA
def sets(model, nv):
    ##season types - set only has one season if steady state model is being used
    if pinp.general['steady_state']:
        model.s_season_types = Set(initialize=[pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]], doc='season types')
    else:
        model.s_season_types = Set(initialize=pinp.general['i_z_idx'][pinp.general['i_mask_z']], doc='season types') #mask season types by the ones included

    #######################
    #labour               #
    #######################
    ##labour periods
    model.s_labperiods = Set(initialize=per.p_date2_df().index, doc='labour periods')

    ##worker levels - levels for the different jobs
    model.s_worker_levels  = Set(initialize=sinp.general['worker_levels'], doc='worker levels for the different jobs')


    #######################
    #cash                 #
    #######################
    ##cashflow periods
    model.s_cashflow_periods = Set(initialize=sinp.general['cashflow_periods'], doc='cashflow periods')

    #######################
    #stubble              #
    #######################
    #stubble categories -  ordered so to allow transferring between categories
    model.s_stub_cat = Set(ordered=True, initialize=pinp.stubble['stub_cat_idx'], doc='stubble categories')

    #######################
    #cropping related     #
    #######################
    #grain pools ie firsts and seconds
    model.s_grain_pools = Set(initialize=sinp.general['grain_pools'], doc='grain pools')

    #landuses that are harvested - used in harv constraints and variables
    model.s_harvcrops = Set(initialize=uinp.mach_general['contract_harvest_speed'].index, doc='landuses that are harvest')

    ##landuses that produce hay - used in hay constraints
    model.s_haycrops = Set(ordered=False, initialize=sinp.landuse['Hay'], doc='landuses that make hay')

    ##types of crops
    model.s_crops = Set(ordered=False, initialize=sinp.landuse['C'], doc='crop types')


    ##all crops and each pasture landuse eg t, tr
    model.s_landuses = Set(ordered=False, initialize=sinp.landuse['All'], doc='landuses')

    ##different fert options - used in labourcroppyomo
    model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')

    ###########
    #rotation #
    ###########

    ##lmus
    lmu_mask = pinp.general['lmu_area'].squeeze() > 0
    model.s_lmus = Set(initialize=pinp.general['lmu_area'].index[lmu_mask],doc='defined the soil type a given rotation is on')

    ##phases
    model.s_phases = Set(initialize=sinp.f_phases().index,doc='rotation phases set')

    ##rotation con1 set
    s_rotcon1 = pd.read_excel('Rotation.xlsx',sheet_name='rotation con1 set',header=None,index_col=0,engine='openpyxl')
    model.s_rotconstraints = Set(initialize=s_rotcon1.index,doc='rotation constraints histories')

    # ##phases disaggregated - used in rot yield transfer
    # def phases_dis():
    #     phase=sinp.stock['phases'].copy()
    #     return phase.set_index(list(range(sinp.general['phase_len']))).index
    # model.s_phases_dis = Set(dimen=sinp.general['phase_len'], ordered=True, initialize=phases_dis(), doc='rotation phases disaggregated')
    # model.s_phases_dis.pprint()

    # model.s_rotconstraints.pprint()

    ##con2 set
    # s_rotcon2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2 set', header= None, index_col = 0)
    # model.s_rotconstraints2 = Set(initialize=s_rotcon2.index, doc='rotation constraints histories 2')



    #######################
    #mvf                  #
    #######################
    model.s_mvf_q = Set(initialize=pinp.mvf['i_q_idx'], doc='mvf levels')

    #######################
    #sheep                #
    #######################
    '''other sheep sets are built in stockpyomo.py'''
    ##all groups
    model.s_infrastructure = Set(initialize=uinp.sheep['i_h1_idx'], doc='core sheep infrastructure')

    ##feed pool
    keys_nv = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    model.s_feed_pools = Set(initialize=keys_nv, doc='nutritive value pools')

    ##dams
    model.s_nut_dams = Set(initialize=np.array(['n%s'%i for i in range(sinp.structuralsa['i_n1_matrix_len'])]), doc='Nutrition levels in each feed period for dams')

    ##offs
    model.s_sale_offs = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t3_len'])], doc='Sales within the year for offs')
    model.s_nut_offs = Set(initialize=np.array(['n%s'%i for i in range(sinp.structuralsa['i_n3_matrix_len'])]), doc='Nutrition levels in each feed period for offs')

    ##prog
    model.s_sale_prog = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t2_len'])], doc='Sales and transfers options for yatf')
    model.s_lw_prog = Set(initialize=['w%03d'%i for i in range(sinp.structuralsa['i_progeny_w2_len'])], doc='Standard LW patterns prog')


    #######################
    #pasture             #
    #######################
    ##pasture types
    model.s_pastures = Set(initialize=sinp.general['pastures'][pinp.general['pas_inc']],doc='feed periods')

    ##feed periods
    model.s_feed_periods = Set(ordered=True, initialize=pinp.period['i_fp_idx'], doc='feed periods') #must be ordered so it can be sliced in pasture pyomo to allow feed to be transferred between periods.

    ##pasture groups
    model.s_dry_groups = Set(initialize=sinp.general['dry_groups'], doc='dry feed pools')
    model.s_grazing_int = Set(initialize=sinp.general['grazing_int'], doc='grazing intensity in the growth/grazing activities')
    model.s_foo_levels = Set(initialize=sinp.general['foo_levels'], doc='FOO level in the growth/grazing activities')




















