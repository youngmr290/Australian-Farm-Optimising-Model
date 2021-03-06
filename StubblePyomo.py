# -*- coding: utf-8 -*-
"""
author: young
"""

#python modules
import pyomo.environ as pe

#AFO modules
import Stubble as stub
import PropertyInputs as pinp

def stub_precalcs(params, report, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    stub.stubble_all(params, report, nv)
    
    
    
def stubpyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''
    ###################
    # variable         #
    ###################
    ##stubble consumption
    model.v_stub_con = pe.Var(model.s_feed_pools,model.s_feed_periods,model.s_crops,model.s_stub_cat,bounds=(0.0,None),
                              doc='consumption of stubble')
    ##stubble transfer
    model.v_stub_transfer = pe.Var(model.s_feed_periods,model.s_crops,model.s_stub_cat,bounds=(0.0,None),
                                   doc='transfer of 1t of stubble to following period')

    ####################
    #define parameters #
    ####################
    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]
    model.p_harv_prop = pe.Param(model.s_feed_periods, model.s_crops, initialize=params[season]['cons_prop'], default = 0.0, mutable=False, doc='proportion of the way through each fp harvest occurs (0 if harv doesnt occur in given period)')
    
    model.p_stub_md = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_crops, model.s_stub_cat, initialize=params[season]['md'], default = 0.0, mutable=False, doc='md from 1t of each stubble categories for each crop')

    model.p_stub_vol = pe.Param(model.s_feed_periods, model.s_crops, model.s_stub_cat, initialize=params[season]['vol'], default = 0.0, mutable=False, doc='amount of intake volume required by 1t of each stubble category for each crop')
    
    model.p_a_req = pe.Param(model.s_feed_periods, model.s_crops, model.s_stub_cat, initialize=params[season]['cat_a_st_req'], default = 0.0, mutable=False, doc='stubble required in each feed periods in order to consume 1t of cat A')
    
    model.p_bc_prov = pe.Param(model.s_crops, model.s_stub_cat, initialize=params['transfer_prov'], default = 0.0, doc='stubble B provided from 1t of cat A and stubble C provided from 1t of cat B')
    
    model.p_bc_req = pe.Param(model.s_crops, model.s_stub_cat, initialize=params['transfer_req'], default = 0.0, doc='stubble required from the row inorder to consume cat B or cat C')
    
    model.p_fp_transfer = pe.Param(model.s_feed_periods, model.s_crops, initialize=params[season]['per_transfer'], default = 0.0, mutable=False, doc='stubble cat B or cat C transferred to the next feed period')
    

    ########################
    #call local constraint #
    ########################
    f_con_stubble_bcd(model)



###################
#local constraint #
###################
def f_con_stubble_bcd(model):
    ''' Links the consumption of a given category with the provision of another category or the transfer of
    stubble to the following period. Eg category A consumption provides category B. Category B can either be
    consumed (hence providing cat C) or transferred to the following period.
    '''
    ##stubble transter from category to category and period to period
    def stubble_transfer(model,f,k,s):
        if s == 'a':# or model.p_bc_req[k,s]==0: #this constraint is only for cat b and c
            return pe.Param.Skip
        else:
            ss = list(model.s_stub_cat)[list(model.s_stub_cat).index(s)-1] #previous stubble cat - used to transfer from current cat to the next, list is required because indexing of an ordered set starts at 1 which means index of 0 chucks error
            fs = list(model.s_feed_periods)[list(model.s_feed_periods).index(f)-1] #have to convert to a list first because indexing of an ordered set starts at 1
            return  - model.v_stub_transfer[fs,k,s] * model.p_fp_transfer[f,k]  + model.v_stub_transfer[f,k,s] * 1000 \
                    + sum(-model.v_stub_con[v,f,k,ss] * model.p_bc_prov[k,s] + model.v_stub_con[v,f,k,s] * model.p_bc_req[k,s] for v in model.s_feed_pools) <=0
    model.con_stubble_bcd = pe.Constraint(model.s_feed_periods, model.s_crops, model.s_stub_cat, rule = stubble_transfer, doc='links rotation stubble production with consumption of cat A')

###################
#constraint global#
###################
##stubble transter from category to category and period to period
def stubble_req_a(model,k,s):
    '''
    Calculate the total stubble required to consume the selected volume category A stubble in each period.

    Used in global constraint (con_stubble_a). See CorePyomo
    '''

    return sum(model.v_stub_con[v,f,k,s] * model.p_a_req[f,k,s] for v in model.s_feed_pools for f in model.s_feed_periods if pe.value(model.p_a_req[f,k,s]) !=0)


##stubble md
def stubble_me(model,p6,f):
    '''
    Calculate the total energy provided to each nv pool by feed the selected amount of stubble.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_stub_con[f,p6,k,s] * model.p_stub_md[f,p6,k,s] for k in model.s_crops for s in model.s_stub_cat)
    
##stubble vol
def stubble_vol(model,p6,f):
    '''
    Calculate the total volume required by each nv pool to feed the selected amount of stubble.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_stub_con[f,p6,k,s] * model.p_stub_vol[p6,k,s] for k in model.s_crops for s in model.s_stub_cat)