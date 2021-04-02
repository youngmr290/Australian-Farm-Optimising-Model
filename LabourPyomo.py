# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:55:51 2019

module: labour pyomo module - contains pyomo params, variables and constraints

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

@author: young
"""

#python modules
from pyomo.environ import *

#AFO modules
import Labour as lab
from CreateModel import *
import PropertyInputs as pinp

def lab_precalcs(params, r_vals):
    lab.labour_general(params, r_vals)
    lab.perm_cost(params, r_vals)
    lab.manager_cost(params, r_vals)
    params['min_perm'] = pinp.labour['min_perm'] 
    params['max_perm'] = pinp.labour['max_perm']
    params['min_managers'] = pinp.labour['min_managers'] 
    params['max_managers'] = pinp.labour['max_managers']
                    


def labpyomo_local(params):
    ############
    # variable  #
    ############

    # Casual supervision
    try:
        model.del_component(model.v_casualsupervision_perm)
    except AttributeError:
        pass
    model.v_casualsupervision_perm = Var(model.s_labperiods,bounds=(0,None),
                                         doc='hours of perm labour used for supervision of casual')
    try:
        model.del_component(model.v_casualsupervision_manager)
    except AttributeError:
        pass
    model.v_casualsupervision_manager = Var(model.s_labperiods,bounds=(0,None),
                                            doc='hours of manager labour used for supervision of casual')

    # Amount of casual. Casual labour can be optimised for each period
    try:
        model.del_component(model.v_quantity_casual)
    except AttributeError:
        pass
    model.v_quantity_casual = Var(model.s_labperiods,bounds=(0,None),
                                  doc='number of casual labour used in each labour period')

    # Amount of permanent labour.
    try:
        model.del_component(model.v_quantity_perm)
    except AttributeError:
        pass
    max_perm = pinp.labour['max_perm'] if pinp.labour['max_perm'] != 'inf' else None  # if none convert to python None
    model.v_quantity_perm = Var(bounds=(pinp.labour['min_perm'],max_perm),
                                doc='number of permanent labour used in each labour period')

    # Amount of manager labour
    try:
        model.del_component(model.v_quantity_manager)
    except AttributeError:
        pass
    max_managers = pinp.labour['max_managers'] if pinp.labour[
                                                      'max_managers'] != 'inf' else None  # if none convert to python None
    model.v_quantity_manager = Var(bounds=(pinp.labour['min_managers'],max_managers),
                                   doc='number of manager/owner labour used in each labour period')

    # manager pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_sheep_labour_manager_index)
        model.del_component(model.v_sheep_labour_manager)
    except AttributeError:
        pass
    model.v_sheep_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                       doc='manager labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_crop_labour_manager_index)
        model.del_component(model.v_crop_labour_manager)
    except AttributeError:
        pass
    model.v_crop_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='manager labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_fixed_labour_manager_index)
        model.del_component(model.v_fixed_labour_manager)
    except AttributeError:
        pass
    model.v_fixed_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                       doc='manager labour used by fixed activities in each labour period for each different worker level')

    # permanent pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_sheep_labour_permanent_index)
        model.del_component(model.v_sheep_labour_permanent)
    except AttributeError:
        pass
    model.v_sheep_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                         doc='permanent labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_crop_labour_permanent_index)
        model.del_component(model.v_crop_labour_permanent)
    except AttributeError:
        pass
    model.v_crop_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                        doc='permanent labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_fixed_labour_permanent_index)
        model.del_component(model.v_fixed_labour_permanent)
    except AttributeError:
        pass
    model.v_fixed_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                         doc='permanent labour used by fixed activities in each labour period for each different worker level')

    # casual pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_sheep_labour_casual_index)
        model.del_component(model.v_sheep_labour_casual)
    except AttributeError:
        pass
    model.v_sheep_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='casual labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_crop_labour_casual_index)
        model.del_component(model.v_crop_labour_casual)
    except AttributeError:
        pass
    model.v_crop_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                     doc='casual labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    try:
        model.del_component(model.v_fixed_labour_casual_index)
        model.del_component(model.v_fixed_labour_casual)
    except AttributeError:
        pass
    model.v_fixed_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='casual labour used by fixed activities in each labour period for each different worker level')


    #########
    #param  #
    #########

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    ##called here , used below to generate params
    try:
        model.del_component(model.p_perm_hours)
    except AttributeError:
        pass
    model.p_perm_hours = Param(model.s_labperiods, initialize= params[season]['permanent hours'], mutable=True, doc='hours worked by a permanent staff in each period')
    
    try:
        model.del_component(model.p_perm_supervision)
    except AttributeError:
        pass
    model.p_perm_supervision = Param(model.s_labperiods, initialize= params[season]['permanent supervision'], mutable=True, doc='hours of supervision required by a permanent staff in each period')
    
    try:
        model.del_component(model.p_perm_cost)
    except AttributeError:
        pass
    model.p_perm_cost = Param(model.s_cashflow_periods, initialize = params['perm_cost'], default = 0.0, doc = 'cost of a permanent staff for 1 yr')
    
    try:
        model.del_component(model.p_casual_cost_index)
        model.del_component(model.p_casual_cost)
    except AttributeError:
        pass
    model.p_casual_cost = Param(model.s_labperiods, model.s_cashflow_periods,  initialize = params[season]['casual_cost'], mutable=True, default = 0.0, doc = 'cost of a casual staff for each labour period')
    
    try:
        model.del_component(model.p_casual_hours)
    except AttributeError:
        pass
    model.p_casual_hours = Param(model.s_labperiods, initialize= params[season]['casual hours'], mutable=True, doc='hours worked by a casual staff in each period')
    
    try:
        model.del_component(model.p_casual_supervision)
    except AttributeError:
        pass
    model.p_casual_supervision = Param(model.s_labperiods, initialize= params[season]['casual supervision'], mutable=True, doc='hours of supervision required by a casual staff in each period')
    
    try:
        model.del_component(model.p_manager_hours)
    except AttributeError:
        pass
    model.p_manager_hours = Param(model.s_labperiods, initialize= params[season]['manager hours'], mutable=True, doc='hours worked by a manager in each period')
    
    try:
        model.del_component(model.p_manager_cost)
    except AttributeError:
        pass
    model.p_manager_cost = Param(model.s_cashflow_periods, initialize = params['manager_cost'], doc = 'cost of a manager for 1 yr')
    
    try:
        model.del_component(model.p_casual_upper)
    except AttributeError:
        pass
    model.p_casual_upper = Param(model.s_labperiods, initialize = params[season]['casual ub'], mutable=True,  doc = 'casual availability upper bound')
    
    try:
        model.del_component(model.p_casual_lower)
    except AttributeError:
        pass
    model.p_casual_lower = Param(model.s_labperiods, initialize = params[season]['casual lb'], mutable=True, doc = 'casual availability lower bound')

    ###############################
    #local constraints            #
    ###############################
    #to constrain the amount of casual labour in each period
    #this can't be done with variable bounds because it's not a constant value for each period (seeding and harv may differ)
    try:
        model.del_component(model.con_casual_bounds)
    except AttributeError:
        pass
    def casual_labour_availability(model, p):
        return  (model.p_casual_lower[p], model.v_quantity_casual[p], model.p_casual_upper[p]) #pyomos way of: lower <= x <= upper
    model.con_casual_bounds = Constraint(model.s_labperiods, rule = casual_labour_availability, doc='bounds the casual labour in each period')
    
    ##casual supervision - can be done by either perm or manager
    try:
        model.del_component(model.con_casual_supervision)
    except AttributeError:
        pass
    def transfer_casual_supervision(model,p):
        return -model.v_casualsupervision_manager[p] - model.v_casualsupervision_perm[p] + (model.p_casual_supervision[p] * model.v_quantity_casual[p]) <= 0
    model.con_casual_supervision = Constraint(model.s_labperiods, rule = transfer_casual_supervision, doc='casual require supervision from perm or manager')

    #manager, this is a little more complex because also need to subtract the supervision hours off of the manager supply of workable hours
    try:
        model.del_component(model.con_labour_transfer_manager)
    except AttributeError:
        pass
    def labour_transfer_manager(model,p):
        return -(model.v_quantity_manager * model.p_manager_hours[p]) + (model.v_quantity_perm * model.p_perm_supervision[p]) + model.v_casualsupervision_manager[p]      \
        + sum(model.v_sheep_labour_manager[p,w] + model.v_crop_labour_manager[p,w] + model.v_fixed_labour_manager[p,w] for w in model.s_worker_levels)  <= 0
    model.con_labour_transfer_manager = Constraint(model.s_labperiods, rule = labour_transfer_manager, doc='labour from manager to sheep and crop and fixed')

    #permanent 
    try:
        model.del_component(model.con_labour_transfer_permanent)
    except AttributeError:
        pass
    def labour_transfer_permanent(model,p):
        return -(model.v_quantity_perm * model.p_perm_hours[p]) + model.v_casualsupervision_perm[p]  \
        + sum(model.v_sheep_labour_permanent[p,w] + model.v_crop_labour_permanent[p,w] + model.v_fixed_labour_permanent[p,w] for w in model.s_worker_levels if w in sinp.general['worker_levels'][0:-1]) <= 0 #if statement just to remove unnecessary activities from lp output
    model.con_labour_transfer_permanent = Constraint(model.s_labperiods, rule = labour_transfer_permanent, doc='labour from permanent staff to sheep and crop and fixed')
    
    #casual note perm and manager can do casual tasks - variables may need to change name so to be less confusing
    try:
        model.del_component(model.con_labour_transfer_casual)
    except AttributeError:
        pass
    def labour_transfer_casual(model,p):
        return -(model.v_quantity_casual[p] *  model.p_casual_hours[p])  \
            + sum(model.v_sheep_labour_casual[p,w] + model.v_crop_labour_casual[p,w] + model.v_fixed_labour_casual[p,w] for w in model.s_worker_levels if w in sinp.general['worker_levels'][0])  <= 0  #if statement just to remove unnecessary activities from lp output
    model.con_labour_transfer_casual = Constraint(model.s_labperiods, rule = labour_transfer_casual, doc='labour from casual staff to sheep and crop and fixed')



#######################
#labour cost function #
#######################

#sum the cost of perm, casual and manager labour. When i tried to do it all in one function it didn't work (it should be possible though )
def casual(model,c):
    return sum( model.v_quantity_casual[p] * model.p_casual_cost[p,c] for p in model.s_labperiods) 
def perm(model,c):
    return model.v_quantity_perm * model.p_perm_cost[c] 
def manager(model,c):
    return model.v_quantity_manager * model.p_manager_cost[c] 
def labour_cost(model,c):
    return casual(model,c) + perm(model,c) + manager(model,c)





