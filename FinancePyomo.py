"""

author: young

"""
#python modules
from pyomo.environ import *

#AFO modules
import Finance as fin
import PropertyInputs as pinp

def fin_precalcs(params, r_vals):
    '''
    Call finance precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    fin.overheads(params, r_vals)
    fin.finance_rep(r_vals)
    params['overdraw'] = pinp.finance['overdraw_limit']


def finpyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    ############
    #variables #
    ############

    #credit for a given time period (time period defined by cashflow set)
    model.v_credit = Var(model.s_cashflow_periods, bounds = (0.0, None), doc = 'amount of net positive cashflow in a given period')
    #debit for a given time period (time period defined by cashflow set)
    model.v_debit = Var(model.s_cashflow_periods, bounds = (0.0, None), doc = 'amount of net negative cashflow in a given period')
    ##dep
    model.v_dep = Var(bounds = (0.0, None), doc = 'transfers total dep to objective')
    ##dep
    model.v_asset = Var(bounds = (0.0, None), doc = 'transfers total value of asset to objective to ensure opportunity cost is represented')
    ##minroe
    model.v_minroe = Var(bounds = (0.0, None), doc = 'total expenditure, used to ensure min return is met')

    ####################
    #params            #
    ####################
    model.p_overhead_cost = Param(model.s_cashflow_periods, initialize = params['overheads'], doc = 'cost of overheads each period')

    #########################
    #call Local constrain   #
    #########################
    f_con_overdraw(params, model)



############
#Contraints#
############
def f_con_overdraw(params, model):
    '''
    Constrains the level of overdraw in each cashflow period.

    This ensures the model draws a realistic level of money from the bank. The user can specify the
    maximum overdraw level.
    '''
    ##debit can't be more than a specified amount ie farmers will draw a maximum from the bank throughout yr
    def overdraw(model,c): 
        return model.v_debit[c] <= params['overdraw']
    model.con_overdraw = Constraint(model.s_cashflow_periods, rule=overdraw, doc='overdraw limit')

