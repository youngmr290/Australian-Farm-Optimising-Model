# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:40:57 2019


module: core module - contains constraints for core equations and objective

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
     

@author: young
"""
import time 
import pyomo.environ as pe
import numpy as np
import networkx
# import pyomo.pysp.util.rapper as rapper
# import pyomo.pysp.plugins.csvsolutionwriter as csvw
# import pyomo.pysp.plugins.jsonsolutionwriter as jsonw
import os
import shutil

#AFO modules - should only be pyomo modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
import CropPyomo as crppy
import MachPyomo as macpy
import LabourPyomo as labpy
import LabourCropPyomo as lcrppy
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
import StockPyomo as stkpy
import MVF as mvf
import Sensitivity as sen
 
import Finance as fin


def coremodel_all(params, trial_name, model):
    '''
    Wraps all of the core model into a function so it can be run multiple times in a loop

    Returns
    -------
    None.

    '''
    #######################################################################################################################################################
    #######################################################################################################################################################
    #core model (pyomo constraints)
    #######################################################################################################################################################
    #######################################################################################################################################################
    
    ######################
    #Labour fixed        #
    ######################
    ##Fixed labour jobs that can be completed by anyone ie this constraint links labour fixed casual and perm and manager supply and requirement.
    def labour_fixed_casual(model,p,w):
        return -model.v_fixed_labour_casual[p,w] - model.v_fixed_labour_permanent[p,w] - model.v_fixed_labour_manager[p,w] +  model.p_super_labour[p] + model.p_tax_labour[p] + model.p_bas_labour[p] <= 0
    model.con_labour_fixed_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_fixed_casual, doc='link between labour supply and requirement by fixed jobs for casual and above')
    
    ##Fixed labour jobs that must be completed by the manager ie this constraint links labour fixed manager supply and requirement.
    def labour_fixed_manager(model,p,w):
        return -model.v_fixed_labour_manager[p,w] +  model.p_planning_labour[p] + (model.p_learn_labour * model.v_learn_allocation[p]) <= 0
    model.con_labour_fixed_manager = pe.Constraint(model.s_labperiods, ['mngr'], rule = labour_fixed_manager, doc='link between labour supply and requirement by fixed jobs for manager')
    
    ######################
    #Labour crop         #
    ######################
    ##labour crop - can be done by anyone
    def labour_crop_anyone(model,p,w):
        return -model.v_crop_labour_casual[p,w] - model.v_crop_labour_permanent[p,w] - model.v_crop_labour_manager[p,w] + lcrppy.mach_labour_anyone(model,p) <= 0
    model.con_labour_crop_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_crop_anyone, doc='link between labour supply and requirement by crop jobs for all labour sources')
    
    ##labour crop - can be done by perm and manager
    def labour_crop_perm(model,p,w):
        return - model.v_crop_labour_permanent[p,w] - model.v_crop_labour_manager[p,w] + lcrppy.mach_labour_perm(model,p) <= 0
    model.con_labour_crop_perm = pe.Constraint(model.s_labperiods, ['perm'], rule = labour_crop_perm, doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')

    ######################
    #labour Sheep        #
    ######################
    ##labour sheep - can be done by anyone
    def labour_sheep_cas(model,p,w):
        return -model.v_sheep_labour_casual[p,w] - model.v_sheep_labour_permanent[p,w] - model.v_sheep_labour_manager[p,w] + suppy.sup_labour(model,p) + stkpy.stock_labour_anyone(model,p) <= 0
    model.con_labour_sheep_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_sheep_cas, doc='link between labour supply and requirement by sheep jobs for all labour sources')

    ##labour sheep - can be done by permanent and manager staff
    def labour_sheep_perm(model,p,w):
        return - model.v_sheep_labour_permanent[p,w] - model.v_sheep_labour_manager[p,w] + stkpy.stock_labour_perm(model,p) <= 0
    model.con_labour_sheep_perm = pe.Constraint(model.s_labperiods, ['perm'], rule = labour_sheep_perm, doc='link between labour supply and requirement by sheep jobs for perm labour sources')

    ##labour sheep - can be done by manager
    def labour_sheep_manager(model,p,w):
        return  - model.v_sheep_labour_manager[p,w] + stkpy.stock_labour_manager(model,p)   <= 0
    model.con_labour_sheep_manager = pe.Constraint(model.s_labperiods, ['mngr'], rule = labour_sheep_manager, doc='link between labour supply and requirement by sheep jobs for manager labour sources')

    #######################################
    #stubble & nap consumption at harvest #
    #######################################
    def harv_stub_nap_cons(model,f):
        if any(model.p_nap_prop[f] or model.p_harv_prop[f,k] for k in model.s_crops):
            return sum(-paspy.pas_me(model,v,f) + sum(model.p_harv_prop[f,k]/(1-model.p_harv_prop[f,k]) * model.v_stub_con[v,f,k,s] * model.p_stub_md[v,f,k,s] for k in model.s_crops for s in model.s_stub_cat)
                    +  model.p_nap_prop[f]/(1-model.p_nap_prop[f]) * paspy.nappas_me(model,v,f) for v in model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip
    model.con_harv_stub_nap_cons = pe.Constraint(model.s_feed_periods, rule = harv_stub_nap_cons, doc='limit stubble and nap consumption in the period harvest occurs')

    ######################
    #stubble             #
    ###################### 
  
    def stubble_a(model,k,s):
        if model.p_rot_stubble[k,s] !=0:
            return   -crppy.rot_stubble(model,k,s) + macpy.stubble_penalty(model,k,s) + stubpy.stubble_req_a(model,k,s) <= 0
        else:
            return pe.Constraint.Skip
    model.con_stubble_a = pe.Constraint(model.s_crops, model.s_stub_cat, rule = stubble_a, doc='links rotation stubble production with consumption of cat A')
    
    ######################
    #sow landuse        #
    ###################### 
   
    ##links crop sow req with mach sow provide - no p set because model can optimise crop sowing time
    def cropsow_link(model,k,l):
        if type(crppy.cropsow(model,k,l)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no crop is being sown on given rotation
        else:
            return sum(-model.v_seeding_crop[p,k,l] for p in model.s_labperiods) + crppy.cropsow(model,k,l)  <= 0
    model.con_cropsow = pe.Constraint(model.s_crops, model.s_lmus, rule = cropsow_link, doc='link between mach sow provide and rotation crop sow require')
   
    ##links pasture sow req with mach sow provide - requires a p set because the timing of sowing pasture is not optimisable (pasture sowing can occur in any period so the user specifies the periods when a given pasture must be sown)
    ##pasture sow has separate constraint from crop rotations because pas sow has a p axis so that user can specify period when pasture is sown (pasture has no yield penalty so model doesnt optimise seeding time like it does for crop)
    def passow_link(model,p,k,l):
        if type(paspy.passow(model,p,k,l)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no pasture is being sown
        else:
            return -model.v_seeding_pas[p,k,l]  + paspy.passow(model,p,k,l) <= 0
    model.con_passow = pe.Constraint( model.s_labperiods, model.s_landuses, model.s_lmus, rule = passow_link, doc='link between mach sow provide and rotation pas sow require')

    ######################
    #harvest crops       #
    ###################### 
    ##links crop and mach pyomo together
    def harv(model,k):
        return  -macpy.harv_supply(model,k) + sum(crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  <= 0
    model.con_harv = pe.Constraint(model.s_harvcrops, rule = harv, doc='harvest constraint')


    ######################
    #harvest hay         #
    ###################### 
    ##links crop and mach pyomo together
    def harv(model,k):
        return  sum(-model.v_hay_made + crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  <= 0
    model.con_makehay = pe.Constraint(model.s_haycrops, rule = harv, doc='make hay constraint')

    #############################
    #yield income & transfer    #
    #############################
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint. 
    def grain_transfer(model,g,k):
        return -crppy.rotation_yield_transfer(model,g,k) + macpy.late_seed_penalty(model,g,k) + sum(model.v_sup_con[k,g,v,f]*1000 for v in model.s_feed_pools for f in model.s_feed_periods)\
                - model.v_buy_grain[k,g]*1000 + model.v_sell_grain[k,g]*1000 <=0
    model.con_grain_transfer = pe.Constraint(model.s_grain_pools, model.s_crops, rule=grain_transfer, doc='constrain grain transfer between rotation and sup feeding')
    
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    def grain_income(model,c):
        return sum(model.v_sell_grain[k,g] * model.p_grain_price[k,c,g] - model.v_buy_grain[k,g]* model.p_buy_grain_price[k,c,g] for k in model.s_crops for g in model.s_grain_pools)
    
    ######################
    #feed                #
    ###################### 
    ##green grazing on crop paddock before seeding
    def poc(model,f,l):
        return -macpy.ha_days_pasture_crop_paddocks(model,f,l) * model.p_poc_con[f,l] + sum(model.v_poc[v,f,l] for v in model.s_feed_pools) <=0
    model.con_poc_available = pe.Constraint(model.s_feed_periods, model.s_lmus, rule=poc, doc='constraint between poc available and consumed')

    ######################
    #  ME                #
    ######################
    def me(model,f,v):
        return -paspy.pas_me(model,v,f) - paspy.nappas_me(model,v,f) - suppy.sup_me(model,v,f) - stubpy.stubble_me(model,v,f) \
               + stkpy.stock_me(model,v,f) - mvf.mvf_me(model,v,f) <=0
    model.con_me = pe.Constraint(model.s_feed_periods, model.s_feed_pools, rule=me, doc='constraint between me available and consumed')

    ######################
    #Vol                 #
    ######################
    def vol(model,f,v):
        return paspy.pas_vol(model,v,f) + suppy.sup_vol(model,v,f) + stubpy.stubble_vol(model,v,f) - stkpy.stock_pi(model,v,f) \
               + mvf.mvf_vol(model,v,f) <=0
    model.con_vol = pe.Constraint(model.s_feed_periods, model.s_feed_pools, rule=vol, doc='constraint between me available and consumed')

    ######################
    #cashflow constraints#
    ######################

    def cash_flow(model,i):
        '''
        Returns
        -------
        Constraint
            combines all cashflow functions from each module and includes debit and credit to form constraint.
            for each cashflow period dollar flow must be greater than 0. this is accomplished by taking a loan from the bank (if there is more exp than income) or depositing money in the bank.
            the money withdrawn or deposited in the bank (debit or credit) is then carried over to the next period.
            the debit and credit carried over is multiplied by j because there is no carry over in the first period (there may be a better way to do it though)
            Carryover basically represents interest free cash at the start of the year. It requires cash from ND and provides in JF.

        '''
        c = sinp.general['cashflow_periods']
        ##j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        ##this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period)
        j = [1] * len(c)
        j[0] = 0
        #todo Revisit the interest calculation at some stage because it didn't tally with the back of envelope estimate by $1000
        return (-grain_income(model,c[i]) + crppy.rotation_cost(model,c[i]) + labpy.labour_cost(model,c[i]) + macpy.mach_cost(model,c[i]) + suppy.sup_cost(model,c[i]) + model.p_overhead_cost[c[i]]
                - stkpy.stock_cashflow(model,c[i])
                - model.v_debit[c[i]] + model.v_credit[c[i]]  + model.v_debit[c[i-1]] * fin.debit_interest() - model.v_credit[c[i-1]] * fin.credit_interest() * j[i] #mul by j so that credit in ND doesnt provide into JF otherwise it will be unbounded because it will get interest
                ) <= 0

    model.con_cashflow = pe.Constraint(range(len(model.s_cashflow_periods)), rule=cash_flow, doc='cashflow')

    ######################
    #dep                 #
    ###################### 
    def dep(model):
        return  macpy.total_dep(model) + suppy.sup_dep(model) - model.v_dep <=0
    model.con_dep = pe.Constraint( rule=dep, doc='tallies depreciation from all activities so it can be transferred to objective')
    
    ######################
    #asset               #
    ###################### 
    def asset(model):
        return (suppy.sup_asset(model) + macpy.mach_asset(model) + stkpy.stock_asset(model)) * uinp.finance['opportunity_cost_capital']  \
                - model.v_asset <=0
    model.con_asset = pe.Constraint( rule=asset, doc='tallies asset from all activities so it can be transferred to objective to represent ROE')
    
    ######################
    #Min ROE             #
    ###################### 
    def minroe(model):
        return (sum(crppy.rotation_cost(model,c)  + labpy.labour_cost(model,c) + macpy.mach_cost(model,c) + suppy.sup_cost(model,c) for c in model.s_cashflow_periods) + stkpy.stock_cost(model)) * fin.f_min_roe() \
                - model.v_minroe <=0   
    model.con_minroe = pe.Constraint(rule=minroe, doc='tallies total expenditure to ensure minimum roe is met')
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #objective
    #######################################################################################################################################################
    #######################################################################################################################################################
    '''
    maximise credit in the last period of cashflow (rather than indexing directly with ND$FLOW, i index with the last name in the cashflow periods in case cashflow periods change) 
    minus dep (variable and fixed)
    '''
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        def profit(model):
            c = sinp.general['cashflow_periods']
            i = len(c) - 1 # minus one because index starts from 0
            return model.v_credit[c[i]]-model.v_debit[c[i]] - model.v_dep - model.v_minroe - model.v_asset - (0.1 * sen.sam['GLPK_fix'])  #have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
                                                                                                                    #sen used to tweak model to to fix glpk when not solving.
        model.profit = pe.Objective(rule=profit, sense=pe.maximize)
        # model.profit.pprint()

    else:
        def ComputeFirstStageCost_rule(model):
            expr = 0
            return expr
        model.FirstStageCost = pe.Expression(rule=ComputeFirstStageCost_rule)

        def ComputeSecondStageCost_rule(model):
            c = sinp.general['cashflow_periods']
            i = len(c) - 1 # minus one because index starts from 0
            return model.v_credit[c[i]]-model.v_debit[c[i]] - model.v_dep - model.v_minroe - (model.v_asset * uinp.finance['opportunity_cost_capital'])  #have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
        model.SecondStageCost = pe.Expression(rule=ComputeSecondStageCost_rule)

        #
        # PySP Auto-generated Objective
        #
        # minimize: sum of StageCosts
        #
        # An active scenario objective equivalent to that generated by PySP is
        # included here for informational purposes.

        def total_cost_rule(model):
            return model.SecondStageCost  # model.FirstStageCost +

        model.Total_Cost_Objective = pe.Objective(rule=total_cost_rule,sense=pe.maximize)


    #######################################################################################################################################################
    #######################################################################################################################################################
    #solve
    #######################################################################################################################################################
    #######################################################################################################################################################

    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:

        ##sometimes if there is a bug when solved it is good to write lp here - because the code doesn't run to the other place where lp written
        model.write('Output/test.lp',io_options={'symbolic_solver_labels':True}) #comment this out when not debugging

        ##tells the solver you want duals and rc
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
        model.slack = pe.Suffix(direction=pe.Suffix.IMPORT)
        ##solve - uses cplex if it exists else glpk - tee=True will print out solver information.
        if not shutil.which("cplex") == None:
            ##solve with cplex if it exists
            solver = pe.SolverFactory('cplex')
        else:
            ##solve with glpk
            solver = pe.SolverFactory('glpk')
            solver.options['tmlim'] = 100 #limit solving time to 100sec in case solver stalls.
        solver_result = solver.solve(model, tee=True)  #turn to true for solver output - may be useful for troubleshooting
        try: #to handle infeasible (there is no profit component when infeasible)
            obj = pe.value(model.profit)
        except ValueError:
            obj = 0

    else:
        '''
        Stage allocation notes:
            If a variable is not allocated to a stage end up in the final stage (they can be optimised independently for each season).
            If a variable is allocated to two stages it is constrained in both stages so it is essentially the same as assigning it to just the first stage.
            Each stage needs at least one variable.
            All variables have at most one time series set therefore don't need to worry about variables being allocated to two stages.
            Dvp dates are the same for all seasons thus don't need to deal with the z axis for dvp allocation (for fp & lp we do need to deal with z axis).
            
        Stage allocation process
            Variables that contain a time series set (eg dvp & lp & fp) are automatically allocated to a stage using the code below.
            The other variables are allocated to a stage manually by the user.
        '''
        #todo include some error handling - each variable needs to assigned to stage - give error if that does not happen, this would be good but cant think of a good way to do it since variables can be in multiple stages. Maybe just check that the variable is in any stage and assume if it is in any stage then it is correct
        # buy grain may not be in stage 3, i feel like you retain grain for the year ahead without knowing the type of season. but then the model will just counter by altering sale of grain.


        import Periods as per
        import Functions as fun


        '''
        Detailed tree with customised variable allocation
        
        Tree Description:
            Currently the tree is simplified from how it will finally end up. Currently it has 9 seasons. The stages are
            based on the timing of break of season. There is also a stage from spring to the end (this indicates that at spring you
            can identify each season).
                                    (early spring node)
                    (root node)           |----z0
                        |----ebrk2spr ----|----z1
                        |                 |----z2
            ----root----|                            (med spring node)
                        |            (mbrk node)           |----z3
                        |                 |----mbrk2spr----|----z4
                        |                 |                |----z5
                        |----ebrk2mbrk----|          (late spring node)
                                          |                |----z6
                                          |----lbrk2spr----|----z7
                                                           |----z8
        
        '''

        stage_info = {}
        stage_info['root'] = {}
        stage_info['ebrk2mbrk'] = {}
        stage_info['ebrk2spr'] = {}
        stage_info['mbrk2spr'] = {}
        stage_info['lbrk2spr'] = {}
        stage_info['ebrk_spr2end'] = {}
        stage_info['mbrk_spr2end'] = {}
        stage_info['lbrk_spr2end'] = {}

        ##specify a season which represents the stages. This is required because for example the fp dates are different for each season (note dvps are the same for all seasons so not required for them)
        root_z = 0 #this could be any stage because all seasons have the same period definitions in the root stage.
        ebrk2mbrk_z = -1 #any season with medium or late break
        ebrk2spr_z = 0 #any season with early break
        mbrk2spr_z = 3 #any season with medium
        lbrk2spr_z = -1 #any season with late break
        ebrk_spr2end_z = 0 #any season with early break
        mbrk_spr2end_z = 3 #any season with medium
        lbrk_spr2end_z = -1 #any season with late break

        ##keys - keys are allocated to each stage and then variable containing any of these sets are allocated to that stage.
        keys_p5 = np.array(per.p_date2_df().index).astype('str')
        keys_p6 = pinp.period['i_fp_idx']
        keys_dams = params['stock']['keys_v_dams']
        keys_prog = params['stock']['keys_v_prog']
        keys_offs = params['stock']['keys_v_offs']

        ##date arrays
        fp_p6z = per.f_feed_periods()[:-1,:].astype('datetime64')
        lp_p5z = per.p_date2_df().to_numpy().astype('datetime64[D]')
        dvp1 = fun.f_baseyr(params['stock']['dvp1'], fp_p6z[0,0].astype('datetime64[Y]'))
        dvp3 = fun.f_baseyr(params['stock']['dvp3'], fp_p6z[0,0].astype('datetime64[Y]'))
        prog_born = fun.f_baseyr(params['stock']['date_born_prog'], fp_p6z[0,0].astype('datetime64[Y]'))

        ##stage dates (these continue into the new year eg some may be in the yr of 2020)
        root_start = np.minimum(np.datetime64(pinp.crop['dry_seed_start']), np.min(per.f_feed_periods().astype(np.datetime64)[0,:]))
        ebrk_start = fp_p6z[0,ebrk2spr_z]
        mbrk_start = fp_p6z[0,mbrk2spr_z]
        spr_start = fp_p6z[0,lbrk2spr_z] + np.timedelta64(50, 'D') #20 days after late seed

        ##add 1 year to date before season start so that periods before season start get allocated to stages.
        lp_p5z[lp_p5z<root_start] = lp_p5z[lp_p5z<root_start] + np.timedelta64(365, 'D')
        dvp1[dvp1<root_start] = dvp1[dvp1<root_start] + np.timedelta64(365, 'D')
        dvp3[dvp3<root_start] = dvp3[dvp3<root_start] + np.timedelta64(365, 'D')
        prog_born[prog_born<root_start] = prog_born[prog_born<root_start] + np.timedelta64(365, 'D')
        ###fp is slightly more complicated because it increments to a new year however for the late break seasons there may be dates with 2020 that are after the start of season date thus they need to be taken back to base yr.
        fp_mask_p6z = np.logical_and(fp_p6z - np.timedelta64(365, 'D') >= root_start, fp_p6z - np.timedelta64(365, 'D') < fp_p6z[0:1,:])
        fp_p6z[fp_mask_p6z] = fp_p6z[fp_mask_p6z] - np.timedelta64(365, 'D')

        ##allocate time sets into stages
        stage_info['root']['sets'] = []
        stage_info['ebrk2mbrk']['sets'] = []
        stage_info['ebrk2spr']['sets'] = []
        stage_info['mbrk2spr']['sets'] = []
        stage_info['lbrk2spr']['sets'] = []
        stage_info['ebrk_spr2end']['sets'] = []
        stage_info['mbrk_spr2end']['sets'] = []
        stage_info['lbrk_spr2end']['sets'] = []
        ###root - early break
        stage_info['root']['sets'].extend(list(keys_p6[np.logical_and(fp_p6z[:,root_z] >= root_start, fp_p6z[:,root_z] < ebrk_start)]))
        stage_info['root']['sets'].extend(list(keys_p5[np.logical_and(lp_p5z[:,root_z] >= root_start, lp_p5z[:,root_z] < ebrk_start)]))
        stage_info['root']['sets'].extend(list(keys_dams[np.logical_and(dvp1 >= root_start, dvp1 < ebrk_start)]))
        stage_info['root']['sets'].extend(list(keys_offs[np.logical_and(dvp3 >= root_start, dvp3 < ebrk_start)]))
        stage_info['root']['sets'].extend(list(keys_prog[np.logical_and(prog_born >= root_start, prog_born < ebrk_start)]))
        ###early break - med break
        stage_info['ebrk2mbrk']['sets'].extend(list(keys_p6[np.logical_and(fp_p6z[:,ebrk2mbrk_z] >= ebrk_start, fp_p6z[:,ebrk2mbrk_z] < mbrk_start)]))
        stage_info['ebrk2mbrk']['sets'].extend(list(keys_p5[np.logical_and(lp_p5z[:,ebrk2mbrk_z] >= ebrk_start, lp_p5z[:,ebrk2mbrk_z] < mbrk_start)]))
        stage_info['ebrk2mbrk']['sets'].extend(list(keys_dams[np.logical_and(dvp1 >= ebrk_start, dvp1 < mbrk_start)]))
        stage_info['ebrk2mbrk']['sets'].extend(list(keys_offs[np.logical_and(dvp3 >= ebrk_start, dvp3 < mbrk_start)]))
        stage_info['ebrk2mbrk']['sets'].extend(list(keys_prog[np.logical_and(prog_born >= ebrk_start, prog_born < mbrk_start)]))
        ###early break - spring
        stage_info['ebrk2spr']['sets'].extend(list(keys_p6[np.logical_and(fp_p6z[:,ebrk2spr_z] >= ebrk_start, fp_p6z[:,ebrk2spr_z] < spr_start)]))
        stage_info['ebrk2spr']['sets'].extend(list(keys_p5[np.logical_and(lp_p5z[:,ebrk2spr_z] >= ebrk_start, lp_p5z[:,ebrk2spr_z] < spr_start)]))
        stage_info['ebrk2spr']['sets'].extend(list(keys_dams[np.logical_and(dvp1 >= ebrk_start, dvp1 < spr_start)]))
        stage_info['ebrk2spr']['sets'].extend(list(keys_offs[np.logical_and(dvp3 >= ebrk_start, dvp3 < spr_start)]))
        stage_info['ebrk2spr']['sets'].extend(list(keys_prog[np.logical_and(prog_born >= ebrk_start, prog_born < spr_start)]))
        ###medium break - spring
        stage_info['mbrk2spr']['sets'].extend(list(keys_p6[np.logical_and(fp_p6z[:,mbrk2spr_z] >= mbrk_start, fp_p6z[:,mbrk2spr_z] < spr_start)]))
        stage_info['mbrk2spr']['sets'].extend(list(keys_p5[np.logical_and(lp_p5z[:,mbrk2spr_z] >= mbrk_start, lp_p5z[:,mbrk2spr_z] < spr_start)]))
        stage_info['mbrk2spr']['sets'].extend(list(keys_dams[np.logical_and(dvp1 >= mbrk_start, dvp1 < spr_start)]))
        stage_info['mbrk2spr']['sets'].extend(list(keys_offs[np.logical_and(dvp3 >= mbrk_start, dvp3 < spr_start)]))
        stage_info['mbrk2spr']['sets'].extend(list(keys_prog[np.logical_and(prog_born >= mbrk_start, prog_born < spr_start)]))
        ###late break - spring (you know late break seasons once you know medium break)
        stage_info['lbrk2spr']['sets'].extend(list(keys_p6[np.logical_and(fp_p6z[:,lbrk2spr_z] >= mbrk_start, fp_p6z[:,lbrk2spr_z] < spr_start)]))
        stage_info['lbrk2spr']['sets'].extend(list(keys_p5[np.logical_and(lp_p5z[:,lbrk2spr_z] >= mbrk_start, lp_p5z[:,lbrk2spr_z] < spr_start)]))
        stage_info['lbrk2spr']['sets'].extend(list(keys_dams[np.logical_and(dvp1 >= mbrk_start, dvp1 < spr_start)]))
        stage_info['lbrk2spr']['sets'].extend(list(keys_offs[np.logical_and(dvp3 >= mbrk_start, dvp3 < spr_start)]))
        stage_info['lbrk2spr']['sets'].extend(list(keys_prog[np.logical_and(prog_born >= mbrk_start, prog_born < spr_start)]))
        ###spring - end (don't need logical and because this is the last stage)
        stage_info['ebrk_spr2end']['sets'].extend(list(keys_p6[fp_p6z[:,ebrk_spr2end_z] >= spr_start]))
        stage_info['ebrk_spr2end']['sets'].extend(list(keys_p5[lp_p5z[:,ebrk_spr2end_z] >= spr_start]))
        stage_info['ebrk_spr2end']['sets'].extend(list(keys_dams[dvp1 >= spr_start]))
        stage_info['ebrk_spr2end']['sets'].extend(list(keys_offs[dvp3 >= spr_start]))
        stage_info['ebrk_spr2end']['sets'].extend(list(keys_prog[prog_born >= spr_start]))
        ###spring - end (don't need logical and because this is the last stage)
        stage_info['mbrk_spr2end']['sets'].extend(list(keys_p6[fp_p6z[:,mbrk_spr2end_z] >= spr_start]))
        stage_info['mbrk_spr2end']['sets'].extend(list(keys_p5[lp_p5z[:,mbrk_spr2end_z] >= spr_start]))
        stage_info['mbrk_spr2end']['sets'].extend(list(keys_dams[dvp1 >= spr_start]))
        stage_info['mbrk_spr2end']['sets'].extend(list(keys_offs[dvp3 >= spr_start]))
        stage_info['mbrk_spr2end']['sets'].extend(list(keys_prog[prog_born >= spr_start]))
        ###spring - end (don't need logical and because this is the last stage)
        stage_info['lbrk_spr2end']['sets'].extend(list(keys_p6[fp_p6z[:,lbrk_spr2end_z] >= spr_start]))
        stage_info['lbrk_spr2end']['sets'].extend(list(keys_p5[lp_p5z[:,lbrk_spr2end_z] >= spr_start]))
        stage_info['lbrk_spr2end']['sets'].extend(list(keys_dams[dvp1 >= spr_start]))
        stage_info['lbrk_spr2end']['sets'].extend(list(keys_offs[dvp3 >= spr_start]))
        stage_info['lbrk_spr2end']['sets'].extend(list(keys_prog[prog_born >= spr_start]))

        ##allocate variable into stages using the allocated sets
        stage_info['root']['vars'] = ['v_quantity_perm[*]','v_quantity_manager[*]','v_sire[*]','v_buy_grain[*,*]','v_root_hist[*,*]']
        stage_info['ebrk2mbrk']['vars'] = []
        stage_info['ebrk2spr']['vars'] = ['v_phase_area[*,*]']
        stage_info['mbrk2spr']['vars'] = ['v_phase_area[*,*]']
        stage_info['lbrk2spr']['vars'] = ['v_phase_area[*,*]']
        stage_info['ebrk_spr2end']['vars'] = ['v_credit[*]', 'v_debit[*]', 'v_dep[*]', 'v_asset[*]', 'v_minroe[*]','v_sell_grain[*,*]','v_infrastructure[*]']
        stage_info['mbrk_spr2end']['vars'] = ['v_credit[*]', 'v_debit[*]', 'v_dep[*]', 'v_asset[*]', 'v_minroe[*]','v_sell_grain[*,*]','v_infrastructure[*]']
        stage_info['lbrk_spr2end']['vars'] = ['v_credit[*]', 'v_debit[*]', 'v_dep[*]', 'v_asset[*]', 'v_minroe[*]','v_sell_grain[*,*]','v_infrastructure[*]']

        for stage in stage_info.keys():
            for v in model.component_objects(pe.Var,active=True):
                for index in v:
                    if index==None: #handle variables with no sets (variables with no sets are manually assigned to stages)
                        continue
                    if any(x in tuple(index) for x in stage_info[stage]['sets'])\
                            or any(x == tuple(index) for x in stage_info[stage]['sets']):
                        if isinstance(index, str): #handle just one set
                            a=str(v)+"["+index+"]"
                        else: #handle when multiple sets
                            a=str(v)+str(list(index))

                        stage_info[stage]['vars'].append(a)

        def pysp_scenario_tree_model_callback():
            # Return a NetworkX scenario tree.
            g = networkx.DiGraph()

            ce1 = 'FirstStageCost' #0$ (for some reason i have to include it even though it is 0)
            ce2 = 'SecondStageCost' #obj function - used for the last stage

            ##root
            g.add_node("Root",
                       cost=ce1,
                       variables=stage_info['root']['vars'],
                       derived_variables=[])

            ##ebrk2mbrk
            ce1 = 'FirstStageCost'
            g.add_node("mbrk",
                       cost=ce1,
                       variables=stage_info['ebrk2mbrk']['vars'],
                       derived_variables=[])
            g.add_edge("Root","mbrk",weight=0.666) #todo this will need to be the season proportion input

            ##ebrk2spr
            g.add_node("ESpr",
                       cost=ce1,
                       variables = stage_info['ebrk2spr']['vars'],
                       derived_variables=[])
            g.add_edge("Root","ESpr",weight=0.334) #todo this will need to be the season proportion input

            ##mbrk2spr
            g.add_node("MSpr",
                       cost=ce1,
                       variables=stage_info['mbrk2spr']['vars'],
                       derived_variables=[])
            g.add_edge("mbrk","MSpr",weight=0.5)

            ##lbrk2spr
            g.add_node("LSpr",
                       cost=ce1,
                       variables=stage_info['lbrk2spr']['vars'],
                       derived_variables=[])
            g.add_edge("mbrk","LSpr",weight=0.5)

            ##ebrk_spr2end
            ###z0
            g.add_node("z0",
                       cost=ce2,
                       variables = stage_info['ebrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("ESpr","z0",weight=0.333)
            ###z1
            g.add_node("z1",
                       cost=ce2,
                       variables = stage_info['ebrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("ESpr","z1",weight=0.333)
            ###z2
            g.add_node("z2",
                       cost=ce2,
                       variables = stage_info['ebrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("ESpr","z2",weight=0.334)

            ##mbrk_spr2end
            ###z3
            g.add_node("z3",
                       cost=ce2,
                       variables=stage_info['mbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("MSpr","z3",weight=0.333)
            ###z4
            g.add_node("z4",
                       cost=ce2,
                       variables=stage_info['mbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("MSpr","z4",weight=0.333)
            ###z5
            g.add_node("z5",
                       cost=ce2,
                       variables=stage_info['mbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("MSpr","z5",weight=0.334)

            ##lbrk_spr2end
            ###z6
            g.add_node("z6",
                       cost=ce2,
                       variables=stage_info['lbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("LSpr","z6",weight=0.333)
            ###z7
            g.add_node("z7",
                       cost=ce2,
                       variables=stage_info['lbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("LSpr","z7",weight=0.333)
            ###z8
            g.add_node("z8",
                       cost=ce2,
                       variables=stage_info['lbrk_spr2end']['vars'],
                       derived_variables=[])
            g.add_edge("LSpr","z8",weight=0.334)

            return g






        '''
        simple tree with simple variable allocation
        '''
        # ##option1 - puts all variables into root.
        # root_vars=[]
        # for v in model.component_objects(pe.Var,active=True):
        #     print(v)
        #     try:
        #         index = np.array(v)
        #         if index.ndim == 1:
        #             len_v=1
        #         else:
        #             len_v = len(index[0]) #number of sets in variable.
        #     except:
        #         len_v=1
        #     a=str(v) + "[" + "*,"*(len_v-1) + "*" + "]"
        #     root_vars.append(a)
        #
        # ##option2 - puts all variables into root.
        # root_vars=[]
        # for v in model.component_objects(pe.Var,active=True):
        #     print(v)
        #     for index in v:
        #         print(v,index)
        #         try:
        #             if isinstance(index, str): #handle just one set
        #                 a=str(v)+"["+index+"]"
        #             else: #handle when multiple sets
        #                 a=str(v)+str(list(index))
        #         except: #handle if variable has no sets
        #             a=str(v)+"['']"
        #         root_vars.append(a)

        # ##option 3 - manually assign all variables
        # root_vars = ['v_poc[*,*,*]']
        #
        # stage2_vars=['v_quantity_perm[*]','v_quantity_manager[*]','v_quantity_casual[*]','v_hay_made[*]','v_phase_area[*,*]','v_sell_grain[*,*]',
        #              'v_credit[*]',
        #              'v_debit[*]',
        #              'v_dep[*]',
        #              'v_asset[*]',
        #              'v_minroe[*]',
        #              'v_buy_grain[*,*]',
        #              'v_sup_con[*,*,*,*]',
        #              'v_stub_con[*,*,*,*]',
        #              'v_stub_transfer[*,*,*]',
        #              'v_infrastructure[*]',
        #              'v_seeding_machdays[*,*,*]',
        #              'v_seeding_pas[*,*,*]',
        #              'v_seeding_crop[*,*,*]',
        #              'v_contractseeding_ha[*,*,*]',
        #              'v_harv_hours[*,*]',
        #              'v_contractharv_hours[*]',
        #              'v_learn_allocation[*]',
        #              'v_casualsupervision_perm[*]',
        #              'v_casualsupervision_manager[*]',
        #              'v_sheep_labour_manager[*,*]',
        #              'v_crop_labour_manager[*,*]',
        #              'v_fixed_labour_manager[*,*]',
        #              'v_sheep_labour_permanent[*,*]',
        #              'v_crop_labour_permanent[*,*]',
        #              'v_fixed_labour_permanent[*,*]',
        #              'v_sheep_labour_casual[*,*]',
        #              'v_crop_labour_casual[*,*]',
        #              'v_fixed_labour_casual[*,*]',
        #              'v_greenpas_ha[*,*,*,*,*,*]',
        #              'v_drypas_consumed[*,*,*,*]',
        #              'v_drypas_transfer[*,*,*]',
        #              'v_nap_consumed[*,*,*,*]',
        #              'v_nap_transfer[*,*,*]',
        #              # 'v_poc[*,*,*]',
        #              'v_sire[*]',
        #              'v_dams[*,*,*,*,*,*,*,*,*]',
        #              'v_offs[*,*,*,*,*,*,*,*,*,*,*]',
        #              'v_prog[*,*,*,*,*,*,*,*]'
        #              ]
        #
        # def pysp_scenario_tree_model_callback():
        #     # Return a NetworkX scenario tree.
        #     g = networkx.DiGraph()
        #
        #     ##root
        #     ce1 = 'FirstStageCost'
        #     g.add_node("Root",
        #                cost=ce1,
        #                variables=root_vars,
        #                derived_variables=[])
        #
        #     ce2 = 'SecondStageCost'
        #     g.add_node("z0",
        #                cost=ce2,
        #                variables = stage2_vars,
        #                derived_variables=[])
        #     g.add_edge("Root","z0",weight=0.334)
        #
        #     g.add_node("z3",
        #                cost=ce2,
        #                variables=stage2_vars,
        #                derived_variables=[])
        #     g.add_edge("Root","z3",weight=0.334)
        #
        #     g.add_node("z6",
        #                cost=ce2,
        #                variables=stage2_vars,
        #                derived_variables=[])
        #     g.add_edge("Root","z6",weight=0.332)
        #
        #     return g


        def pysp_instance_creation_callback(scenario_name,node_names):
            instance = model.clone()
            print('cloning: ',scenario_name)
            #todo since we wont be using this method i have converted all parameters back to mutable=false. can be reverted easily using global find and replace
            ##stubble
            instance.p_fp_transfer.store_values(params['stub'][scenario_name]['per_transfer'])
            instance.p_a_req.store_values(params['stub'][scenario_name]['cat_a_st_req'])
            instance.p_stub_vol.store_values(params['stub'][scenario_name]['vol'])
            instance.p_stub_md.store_values(params['stub'][scenario_name]['md'])
            instance.p_harv_prop.store_values(params['stub'][scenario_name]['cons_prop'])

            ##labour
            instance.p_perm_hours.store_values(params['lab'][scenario_name]['permanent hours'])
            instance.p_perm_supervision.store_values(params['lab'][scenario_name]['permanent supervision'])
            instance.p_casual_cost.store_values(params['lab'][scenario_name]['casual_cost'])
            instance.p_casual_hours.store_values(params['lab'][scenario_name]['casual hours'])
            instance.p_casual_supervision.store_values(params['lab'][scenario_name]['casual supervision'])
            instance.p_manager_hours.store_values(params['lab'][scenario_name]['manager hours'])
            instance.p_casual_upper.store_values(params['lab'][scenario_name]['casual ub'])
            instance.p_casual_lower.store_values(params['lab'][scenario_name]['casual lb'])

            ##labour crop
            instance.p_prep_pack.store_values(params['crplab'][scenario_name]['prep_labour'])
            instance.p_fert_app_hour_tonne.store_values(params['crplab'][scenario_name]['fert_app_time_t'])
            instance.p_fert_app_hour_ha.store_values(params['crplab'][scenario_name]['fert_app_time_ha'])
            instance.p_chem_app_lab.store_values(params['crplab'][scenario_name]['chem_app_time_ha'])
            instance.p_variable_crop_monitor.store_values(params['crplab'][scenario_name]['variable_crop_monitor'])
            instance.p_fixed_crop_monitor.store_values(params['crplab'][scenario_name]['fixed_crop_monitor'])

            ##labour fixed
            instance.p_super_labour.store_values(params['labfx'][scenario_name]['super'])
            instance.p_bas_labour.store_values(params['labfx'][scenario_name]['bas'])
            instance.p_planning_labour.store_values(params['labfx'][scenario_name]['planning'])
            instance.p_tax_labour.store_values(params['labfx'][scenario_name]['tax'])

            ##crop
            instance.p_rotation_cost.store_values(params['crop'][scenario_name]['rot_cost'])
            instance.p_rotation_yield.store_values(params['crop'][scenario_name]['rot_yield'])
            instance.p_phasefert.store_values(params['crop'][scenario_name]['fert_req'])

            ##pasture
            instance.p_germination.store_values(params['pas'][scenario_name]['p_germination_flrt'])
            instance.p_foo_grn_reseeding.store_values(params['pas'][scenario_name]['p_foo_grn_reseeding_flrt'])
            instance.p_foo_dry_reseeding.store_values(params['pas'][scenario_name]['p_foo_dry_reseeding_dflrt'])
            instance.p_foo_end_grnha.store_values(params['pas'][scenario_name]['p_foo_end_grnha_goflt'])
            instance.p_foo_start_grnha.store_values(params['pas'][scenario_name]['p_foo_start_grnha_oflt'])
            instance.p_senesce_grnha.store_values(params['pas'][scenario_name]['p_senesce_grnha_dgoflt'])
            instance.p_me_cons_grnha.store_values(params['pas'][scenario_name]['p_me_cons_grnha_vgoflt'])
            instance.p_volume_grnha.store_values(params['pas'][scenario_name]['p_volume_grnha_goflt'])
            instance.p_dry_mecons_t.store_values(params['pas'][scenario_name]['p_dry_mecons_t_vdft'])
            instance.p_dry_volume_t.store_values(params['pas'][scenario_name]['p_dry_volume_t_dft'])
            instance.p_dry_transfer_t.store_values(params['pas'][scenario_name]['p_dry_transfer_t_ft'])
            instance.p_nap.store_values(params['pas'][scenario_name]['p_nap_dflrt'])
            instance.p_nap_prop.store_values(params['pas'][scenario_name]['p_harvest_period_prop'])
            instance.p_phase_area.store_values(params['pas'][scenario_name]['p_phase_area_flrt'])
            instance.p_pas_sow.store_values(params['pas'][scenario_name]['p_pas_sow_plrk'])
            instance.p_poc_vol.store_values(params['pas'][scenario_name]['p_poc_vol_f'])

            ##machine
            instance.p_contractseeding_occur.store_values(params['mach'][scenario_name]['contractseeding_occur'])
            instance.p_seed_days.store_values(params['mach'][scenario_name]['seed_days'])
            instance.p_seeding_cost.store_values(params['mach'][scenario_name]['seeding_cost'])
            instance.p_contract_seeding_cost.store_values(params['mach'][scenario_name]['contract_seed_cost'])
            instance.p_harv_rate.store_values(params['mach'][scenario_name]['harv_rate_period'])
            instance.p_harv_hrs_max.store_values(params['mach'][scenario_name]['max_harv_hours'])
            instance.p_harv_cost.store_values(params['mach'][scenario_name]['harvest_cost'])
            instance.p_contractharv_cost.store_values(params['mach'][scenario_name]['contract_harvest_cost'])
            instance.p_yield_penalty.store_values(params['mach'][scenario_name]['yield_penalty'])
            instance.p_seeding_grazingdays.store_values(params['mach'][scenario_name]['grazing_days'])

            ##sup feed
            instance.p_sup_cost.store_values(params['sup'][scenario_name]['total_sup_cost'])
            instance.p_sup_labour.store_values(params['sup'][scenario_name]['sup_labour'])

            ##stock
            instance.p_nsires_req.store_values(params['stock'][scenario_name]['p_nsire_req_dams'])
            instance.p_nsires_prov.store_values(params['stock'][scenario_name]['p_nsire_prov_sire'])
            instance.p_npw.store_values(params['stock'][scenario_name]['p_npw_dams'])
            instance.p_progprov_dams.store_values(params['stock'][scenario_name]['p_progprov_dams'])
            instance.p_progprov_offs.store_values(params['stock'][scenario_name]['p_progprov_offs'])
            instance.p_numbers_prov_dams.store_values(params['stock'][scenario_name]['p_numbers_prov_dams'])
            instance.p_numbers_provthis_dams.store_values(params['stock'][scenario_name]['p_numbers_provthis_dams'])
            instance.p_numbers_prov_offs.store_values(params['stock'][scenario_name]['p_numbers_prov_offs'])
            instance.p_mei_sire.store_values(params['stock'][scenario_name]['p_mei_sire'])
            instance.p_mei_dams.store_values(params['stock'][scenario_name]['p_mei_dams'])
            instance.p_mei_offs.store_values(params['stock'][scenario_name]['p_mei_offs'])
            instance.p_pi_sire.store_values(params['stock'][scenario_name]['p_pi_sire'])
            instance.p_pi_dams.store_values(params['stock'][scenario_name]['p_pi_dams'])
            instance.p_pi_offs.store_values(params['stock'][scenario_name]['p_pi_offs'])
            instance.p_cashflow_sire.store_values(params['stock'][scenario_name]['p_cashflow_sire'])
            instance.p_cashflow_dams.store_values(params['stock'][scenario_name]['p_cashflow_dams'])
            instance.p_cashflow_prog.store_values(params['stock'][scenario_name]['p_cashflow_prog'])
            instance.p_cashflow_offs.store_values(params['stock'][scenario_name]['p_cashflow_offs'])
            instance.p_cost_sire.store_values(params['stock'][scenario_name]['p_cost_sire'])
            instance.p_cost_dams.store_values(params['stock'][scenario_name]['p_cost_dams'])
            instance.p_cost_offs.store_values(params['stock'][scenario_name]['p_cost_offs'])
            instance.p_asset_sire.store_values(params['stock'][scenario_name]['p_assetvalue_sire'])
            instance.p_asset_dams.store_values(params['stock'][scenario_name]['p_assetvalue_dams'])
            instance.p_asset_offs.store_values(params['stock'][scenario_name]['p_assetvalue_offs'])
            instance.p_lab_anyone_sire.store_values(params['stock'][scenario_name]['p_labour_anyone_sire'])
            instance.p_lab_perm_sire.store_values(params['stock'][scenario_name]['p_labour_perm_sire'])
            instance.p_lab_manager_sire.store_values(params['stock'][scenario_name]['p_labour_manager_sire'])
            instance.p_lab_anyone_dams.store_values(params['stock'][scenario_name]['p_labour_anyone_dams'])
            instance.p_lab_perm_dams.store_values(params['stock'][scenario_name]['p_labour_perm_dams'])
            instance.p_lab_manager_dams.store_values(params['stock'][scenario_name]['p_labour_manager_dams'])
            instance.p_lab_anyone_offs.store_values(params['stock'][scenario_name]['p_labour_anyone_offs'])
            instance.p_lab_perm_offs.store_values(params['stock'][scenario_name]['p_labour_perm_offs'])
            instance.p_lab_manager_offs.store_values(params['stock'][scenario_name]['p_labour_manager_offs'])
            instance.p_infra_sire.store_values(params['stock'][scenario_name]['p_infrastructure_sire'])
            instance.p_infra_dams.store_values(params['stock'][scenario_name]['p_infrastructure_dams'])
            instance.p_infra_offs.store_values(params['stock'][scenario_name]['p_infrastructure_offs'])
            instance.p_dse_sire.store_values(params['stock'][scenario_name]['p_dse_sire'])
            instance.p_dse_dams.store_values(params['stock'][scenario_name]['p_dse_dams'])
            instance.p_dse_offs.store_values(params['stock'][scenario_name]['p_dse_offs'])
            instance.p_cost_purch_sire.store_values(params['stock'][scenario_name]['p_purchcost_sire'])

            return instance
        ##clones model, updates params and create scenario tree
        concrete_tree = pysp_scenario_tree_model_callback()
        stsolver = rapper.StochSolver(None,tree_model=concrete_tree,fsfct=pysp_instance_creation_callback)
        ##creates binding constraints on variables and solves.
        solver_result = stsolver.solve_ef('glpk',tee=False,verbose=True) #convert tee=True to see solver output
        obj = stsolver.root_E_obj()
        # for varname,varval in stsolver.root_Var_solution():  # unfortunately this is only for root
        #     print(varname,str(varval))
        ##write lp file
        stsolver.ef_instance.write('Output/%s.lp' % trial_name,io_options={'symbolic_solver_labels': True})

        ##saves file to csv - not used for anything other than looking at.
        csvw.write_csv_soln(stsolver.scenario_tree,"solutionMRY")
        ##saves file to json - cant change the file name..without changing a pyomo module
        jsonw.JSONSolutionWriter.write('',stsolver.scenario_tree,
                                       'ef')  # i don't know what the first arg does?? it needs to exist but can put any string without changing output

    ##this prints trial name, overall profit and feasibility for each trial
    print("Displaying Solution for trial: %s\n" % trial_name,'-' * 60,'\n%s' % obj)

    ##this check if the solver is optimal - if infeasible or error the model will save a file in Output/infeasible/ directory. This will be accessed in reporting to stop you reporting infeasible trials.
    ##the model will keep running the next trials even if one is infeasible.
    if (solver_result.solver.status == pe.SolverStatus.ok) and (
            solver_result.solver.termination_condition == pe.TerminationCondition.optimal):
        print('OPTIMAL LP SOLUTION FOUND')  # Do nothing when the solution in optimal and feasible
        ###trys to delete the infeasible file because the trial is now optimal
        try:
            os.remove('Output/infeasible/%s.txt' % trial_name)
        except FileNotFoundError:
            pass
    elif (solver_result.solver.termination_condition == pe.TerminationCondition.infeasible):
        print('***INFEASIBLE LP SOLUTION***')
        ###save infeasible file
        with open('Output/infeasible/%s.txt' % trial_name,'w') as f:
            f.write("Solver Status: {0}" .format(solver_result.solver.termination_condition))
    else:  # Something else is wrong - solver may have stalled.
        print('***Solver Status: error (other)***')
        ###save infeasible file
        with open('Output/infeasible/%s.txt' % trial_name,'w') as f:
            f.write("Solver Status: {0}" .format(solver_result.solver.termination_condition))

    return obj