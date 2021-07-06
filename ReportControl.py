"""

Background and usage
---------------------
Reports are generated in a two step process. Firstly, each trial undergoes some ‘within trial’ calculations
(e.g. calc the sale price for a trial), the results are stacked together and stored in a table
(each report has its own table). The resulting table can then undergo some ‘between trial’ calculations
(e.g. graphing profit by sale price).

Tweaking the output of a given report is done in the ReportControl.py. Here the user specifies the report
properties. For example they can specify which axes to report in the table (as either rows or columns in the table)
and which to average or another example. The user can also specify report options. For example, what type of profit
(eg is asset opportunity cost included or not) they want to use in the profit by area curve.

To run execute this module with optional args <processor number> <report number>. If no arguments are passed
the 1 processor is used and the 'default' report group is run.

The trials to report are controlled in exp.xl.
The reports to run for a given report number are controlled in exp.xl.

How to add a report:
--------------------
#. Add the required report variables to the ``r_vals`` dictionary - this is done in precalcs
#. Build a 'within trial' report function in Report.py (if an existing one does not meet your criteria)
#. Build a 'between trial' function (if an existing one does not meet your criteria). This is done
   in the 'Final reports' subsection of ReportFunctions.py
#. Add it to exp.xlsx
#. in ReportControl.py add an empty df to stack results
#. in ReportControl.py build the 'within trial' and 'between trial' sections (this can easily be done by copying
   existing code and making the relevant changes).

.. tip:: When creating r_vals values try and do it in obvious spots so it is easier to understand later.

.. note:: For livestock: livestock is slightly more complicated. If you add a r_val or lp_vars you
    will also need to add it to f_stock_reshape the allows you to get the shape correct
    (remove singleton axis and converts lp_vars from dict to numpy).

.. note:: If reporting dates from a numpy array it is necessary to convert to datetime64[ns] prior to converting to a DataFrame
    For example:
    data_df3 = pd.DataFrame(fvp_fdams.astype('datetime64[ns]'))  # conversion to dataframe only works with this datatype

author: Young
"""

#todo add a second mortality report that the weighted average mortality for the animals selected (to complement the current report that is mortality for each w axis)

import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
import glob
import time

import ReportFunctions as rep
import Functions as fun


#report the clock time that the experiment was started
print(f'Reporting commenced at: {time.ctime()}')
start = time.time()


##read in excel that controls which reports to run and slice for the selected experiment.
## If no arg passed in or the experiment is not set up with custom col in report_run then default col is used
report_run = pd.read_excel('exp.xlsx', sheet_name='Run Report', index_col=[0], header=[0,1], engine='openpyxl')
try:
    exp_group = int(sys.argv[3])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
except:  # in case no arg passed to python
    exp_group = "Default"
try:
    report_run = report_run.loc[:,('Run',exp_group)]
except KeyError:  # in case the experiment is not set up with custom report_run
    report_run = report_run.loc[:,('Run',"Default")]
report_run = report_run.to_frame()
report_run = report_run.droplevel(1, axis=1)

#todo Reports to add:
# 1. report which identifies which sale grid sheep go into. Maybe this can be done using arg sorted on the saleprice array (last step of the sale value function).

def f_report(processor, trials, non_exist_trials):
    '''Function to wrap ReportControl.py so that multiprocessing can be used.'''
    # print('Start processor: {0}'.format(processor))
    # print('Start trials: {0}'.format(trials))
    ##create empty df to stack each trial results into
    stacked_infeasible = pd.DataFrame().rename_axis('Trial')  # name of any infeasible trials
    stacked_non_exist = pd.DataFrame(non_exist_trials).rename_axis('Trial')  # name of any infeasible trials
    stacked_summary = pd.DataFrame()  # 1 line summary of each trial
    stacked_areasum = pd.DataFrame()  # area summary
    stacked_pnl = pd.DataFrame()  # profit and loss statement
    stacked_profitarea = pd.DataFrame()  # profit by land area
    stacked_saleprice = pd.DataFrame()  # sale price
    stacked_salevalue_dams = pd.DataFrame()  # average sale value dams
    stacked_salevalue_offs = pd.DataFrame()  # average sale value offs
    stacked_woolvalue_dams = pd.DataFrame()  # average wool value dams
    stacked_woolvalue_offs = pd.DataFrame()  # average wool value offs
    stacked_saledate_offs = pd.DataFrame()  # offs sale date
    stacked_cfw_dams = pd.DataFrame()  # clean fleece weight dams
    stacked_fd_dams = pd.DataFrame()  # fibre diameter dams
    stacked_cfw_offs = pd.DataFrame()  # clean fleece weight dams
    stacked_fd_offs = pd.DataFrame()  # fibre diameter dams
    stacked_wbe_dams = pd.DataFrame()  # whole body energy content dams
    stacked_wbe_offs = pd.DataFrame()  # whole body energy content offs
    stacked_lw_dams = pd.DataFrame()  # live weight dams (large array with p, e and b axis)
    stacked_ffcfw_dams = pd.DataFrame()  # fleece free conceptus free weight dams (large array with p, e and b axis)
    stacked_nv_dams = pd.DataFrame()  # diet nutritive value for dams (large array with p, e and b axis)
    stacked_ffcfw_yatf = pd.DataFrame()  # fleece free conceptus free weight yatf (large array with p, e and b axis)
    stacked_ffcfw_prog = pd.DataFrame()  # fleece free conceptus free weight prog (large array with p, e and b axis)
    stacked_ffcfw_offs = pd.DataFrame()  # fleece free conceptus free weight offs (large array with p, e and b axis)
    stacked_nv_offs = pd.DataFrame()  # diet nutritive value for offs (large array with p, e and b axis)
    stacked_weanper = pd.DataFrame()  # weaning percent
    stacked_scanper = pd.DataFrame()  # scan percent
    stacked_dry_propn = pd.DataFrame()  # dry ewe proportion
    stacked_lamb_survival = pd.DataFrame()  # lamb survival
    stacked_daily_mei_dams = pd.DataFrame()  # mei dams
    stacked_daily_pi_dams = pd.DataFrame()  # potential intake dams
    stacked_daily_mei_offs = pd.DataFrame()  # mei offs
    stacked_daily_pi_offs = pd.DataFrame()  # potential intake offs
    stacked_numbers_dams = pd.DataFrame()  # numbers dams
    stacked_numbers_dams_p = pd.DataFrame()  # numbers dams with p axis (large array)
    stacked_numbers_prog = pd.DataFrame()  # numbers prog
    stacked_numbers_offs = pd.DataFrame()  # numbers offs
    stacked_numbers_offs_p = pd.DataFrame()  # numbers offs with p axis (large array)
    stacked_mort_dams = pd.DataFrame()  # mort dams with p axis (large array)
    stacked_mort_offs = pd.DataFrame()  # mort offs with p axis (large array)
    stacked_dse_sire = pd.DataFrame()  # dse based on normal weight
    stacked_dse_dams = pd.DataFrame()  # dse based on normal weight
    stacked_dse_offs = pd.DataFrame()  # dse based on normal weight
    stacked_dse1_sire = pd.DataFrame()  # dse based on mei
    stacked_dse1_dams = pd.DataFrame()  # dse based on mei
    stacked_dse1_offs = pd.DataFrame()  # dse based on mei
    stacked_grnfoo = pd.DataFrame()  # green foo
    stacked_dryfoo = pd.DataFrame()  # dry foo
    stacked_napfoo = pd.DataFrame()  # non arable pasture foo
    stacked_grncon = pd.DataFrame()  # green pasture consumed
    stacked_drycon = pd.DataFrame()  # dry pasture consumed
    stacked_napcon = pd.DataFrame()  # non arable pasture feed consumed
    stacked_poccon = pd.DataFrame()  # pasture on crop paddocks feed consumed
    stacked_supcon = pd.DataFrame()  # supplement feed consumed
    stacked_stubcon = pd.DataFrame()  # stubble feed consumed
    stacked_grnnv = pd.DataFrame()  # NV of green pas
    stacked_grndmd = pd.DataFrame()  # dmd of green pas
    stacked_avegrnfoo = pd.DataFrame()  # Average Foo of green pas
    stacked_drynv = pd.DataFrame()  # NV of dry pas
    stacked_drydmd = pd.DataFrame()  # dmd of dry pas
    stacked_avedryfoo = pd.DataFrame()  # Average Foo of dry pas

    ##read in the pickled results
    for trial_name in trials:
        lp_vars,r_vals = rep.load_pkl(trial_name)

        ##handle infeasible trials
        if os.path.isfile('Output/infeasible/{0}.txt'.format(trial_name)):
            stacked_infeasible = stacked_infeasible.append(pd.DataFrame([trial_name]).rename_axis('Trial'))
            lp_vars = fun.f_clean_dict(lp_vars) #if a trial is infeasible or doesnt solve all the lp values are None. This function converts them to 0 so the report can still run.

        ##run report functions
        if report_run.loc['run_summary', 'Run']:
            summary = rep.f_summary(lp_vars,r_vals,trial_name)
            summary.index.name = 'Trial'
            stacked_summary = stacked_summary.append(summary)

        if report_run.loc['run_areasum', 'Run']:
            option = 1
            areasum = rep.f_area_summary(lp_vars, r_vals, option=option)
            areasum = pd.concat([areasum],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_areasum = stacked_areasum.append(areasum)

        if report_run.loc['run_pnl', 'Run']:
            pnl = rep.f_profitloss_table(lp_vars, r_vals)
            pnl = pd.concat([pnl],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_pnl = stacked_pnl.append(pnl)

        if report_run.loc['run_profitarea', 'Run']:
            area_option = 3
            profit_option = 0
            profitarea = pd.DataFrame(index=[trial_name], columns=['area','profit'])
            profitarea.loc[trial_name, 'area'] = rep.f_area_summary(lp_vars,r_vals,area_option)
            profitarea.loc[trial_name,'profit'] = rep.f_profit(lp_vars,r_vals,profit_option)
            stacked_profitarea = stacked_profitarea.append(profitarea)

        if report_run.loc['run_saleprice', 'Run']:
            option = 2
            grid = [0,5,6]
            weight = [22,40,25]
            fs = [2,3,2]
            saleprice = rep.f_price_summary(lp_vars, r_vals, option=option, grid=grid, weight=weight, fs=fs)
            saleprice = pd.concat([saleprice],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_saleprice = stacked_saleprice.append(saleprice)

        if report_run.loc['run_salevalue_dams', 'Run']:
            type = 'stock'
            prod = 'salevalue_k2ctva1nwziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = 1
            keys = 'dams_keys_k2ctvanwziy1g1'
            arith = 1
            index =[3]
            cols =[0,1,2]
            salevalue_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            salevalue_dams = pd.concat([salevalue_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salevalue_dams = stacked_salevalue_dams.append(salevalue_dams)

        if report_run.loc['run_salevalue_offs', 'Run']:
            type = 'stock'
            prod = 'salevalue_k3k5ctvnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = 2
            keys = 'offs_keys_k3k5ctvnwziaxyg3'
            arith = 1
            index = [4, 10]     #DVP, gender
            cols = [2, 12, 3]   #cashflow period, g3, t
            salevalue_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            salevalue_offs = pd.concat([salevalue_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salevalue_offs = stacked_salevalue_offs.append(salevalue_offs)

        if report_run.loc['run_woolvalue_dams', 'Run']:
            type = 'stock'
            prod = 'woolvalue_k2ctva1nwziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = 1
            keys = 'dams_keys_k2ctvanwziy1g1'
            arith = 1
            index =[3]
            cols =[1, 2]
            woolvalue_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            woolvalue_dams = pd.concat([woolvalue_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_woolvalue_dams = stacked_woolvalue_dams.append(woolvalue_dams)

        if report_run.loc['run_woolvalue_offs', 'Run']:
            type = 'stock'
            prod = 'woolvalue_k3k5ctvnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = 2
            keys = 'offs_keys_k3k5ctvnwziaxyg3'
            arith = 1
            index = [4, 10]     #DVP, gender
            cols = [2, 12, 3]   #cashflow period, g3, t
            woolvalue_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            woolvalue_offs = pd.concat([woolvalue_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_woolvalue_offs = stacked_woolvalue_offs.append(woolvalue_offs)

        if report_run.loc['run_saledate_offs', 'Run']:
            type = 'stock'
            prod = 'saledate_k3k5tvnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            keys = 'offs_keys_k3k5tvnwziaxyg3'
            arith = 1
            index = [3, 5]              #DVP, w
            cols = [11, 0, 1, 2, 9]     #g3, dam age, BTRT, t, gender
            saledate_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols).astype('datetime64[D]')
            saledate_offs = pd.concat([saledate_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_saledate_offs = stacked_saledate_offs.append(saledate_offs)

        if report_run.loc['run_cfw_dams', 'Run']:
            type = 'stock'
            prod = 'cfw_hdmob_k2tva1nwziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            keys = 'dams_keys_k2tvanwziy1g1'
            arith = 1
            index =[2]
            cols =[0, 1]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]   #LSLN: NM & 00
            cfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            cfw_dams = pd.concat([cfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_cfw_dams = stacked_cfw_dams.append(cfw_dams)

        if report_run.loc['run_fd_dams', 'Run']:
            type = 'stock'
            prod = 'fd_hdmob_k2tva1nwziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            keys = 'dams_keys_k2tvanwziy1g1'
            arith = 1
            index =[2]
            cols =[0, 1]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            fd_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            fd_dams = pd.concat([fd_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_fd_dams = stacked_fd_dams.append(fd_dams)

        if report_run.loc['run_cfw_offs', 'Run']:
            type = 'stock'
            prod = 'cfw_hdmob_k3k5tvnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            keys = 'offs_keys_k3k5tvnwziaxyg3'
            arith = 1
            index = [3]          #DVP
            cols = [11, 0, 1, 2, 9]  #g3, dam age, BTRT, t, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            cfw_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            cfw_offs = pd.concat([cfw_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_cfw_offs = stacked_cfw_offs.append(cfw_offs)

        if report_run.loc['run_fd_offs', 'Run']:
            type = 'stock'
            prod = 'fd_hdmob_k3k5tvnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            keys = 'offs_keys_k3k5tvnwziaxyg3'
            arith = 1
            index = [3]                 #DVP
            cols = [11, 0, 1, 2, 9]     #g3, dam age, BTRT, t, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            fd_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            fd_offs = pd.concat([fd_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_fd_offs = stacked_fd_offs.append(fd_offs)

        if report_run.loc['run_wbe_dams', 'Run']:
            type = 'stock'
            prod = 'wbe_k2va1nwziyg1'
            na_prod = [1]
            weights = 'dams_numbers_k2tvanwziy1g1'
            keys = 'dams_keys_k2tvanwziy1g1'
            arith = 1
            index = [2]
            cols = [0]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            wbe_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            wbe_dams = pd.concat([wbe_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_wbe_dams = stacked_wbe_dams.append(wbe_dams)

        if report_run.loc['run_wbe_offs', 'Run']:
            type = 'stock'
            prod = 'wbe_k3k5vnwziaxyg3'
            na_prod = [2]
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            keys = 'offs_keys_k3k5tvnwziaxyg3'
            arith = 1
            index = [3]             #DVP
            cols = [11, 0, 1, 2]    #g3, dam age, BTRT, t
            axis_slice = {}
            wbe_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            wbe_offs = pd.concat([wbe_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_wbe_offs = stacked_wbe_offs.append(wbe_offs)

        if report_run.loc['run_lw_dams', 'Run']:
            ##Average dam lw with p, e & b axis. Lw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'lw_dams_k2vpa1e1b1nw8ziyg1'
            na_prod = [1]
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [3,5,6]
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'dams_keys_k2tvpaebnwziy1g1'
            arith = 1
            index = [3]
            cols = [6]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            lw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                     , na_weights=na_weights, prod_weights=prod_weights, den_weights=den_weights
                                     , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            lw_dams = pd.concat([lw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_lw_dams = stacked_lw_dams.append(lw_dams)

        if report_run.loc['run_ffcfw_dams', 'Run']:
            ##Average dam ffcfw with p, e & b axis. ffcfw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'ffcfw_dams_k2vpa1e1b1nw8ziyg1'
            na_prod = [1]
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [3, 5, 6]
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'dams_keys_k2tvpaebnwziy1g1'
            arith = 1
            index = [3]
            cols = [6]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            ffcfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                     , na_weights=na_weights, prod_weights=prod_weights
                                     , den_weights=den_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_dams = pd.concat([ffcfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_dams = stacked_ffcfw_dams.append(ffcfw_dams)

        if report_run.loc['run_nv_dams', 'Run']:
            ##Average dam NV with p, e & b axis. NV is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'nv_dams_k2vpa1e1b1nw8ziyg1'
            na_prod = [1]
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [3,5,6]
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'dams_keys_k2tvpaebnwziy1g1'
            arith = 1
            index = [3]             #p
            cols = [12, 5, 1, 6]    #g1, e, t & b1
            axis_slice = {}
            # axis_slice[5] = [0, 1, 1]   #only the first cycle of e1
            nv_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod,
                                   prod_weights=prod_weights, weights=weights, na_weights=na_weights, den_weights=den_weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            nv_dams = pd.concat([nv_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_nv_dams = stacked_nv_dams.append(nv_dams)

        if report_run.loc['run_ffcfw_yatf', 'Run']:
            ##Average yatf ffcfw with p, e & b axis. ffcfw is not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            ##For yatf the denom weight also includes a weighting for nyatf. The numerator also gets weighted by this.
            ##v_dam must be used because v_prog has a different w axis than yatf.
            type = 'stock'
            prod = 'ffcfw_yatf_k2vpa1e1b1nw8zixyg1'
            na_prod = [1]                               #t
            prod_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [3, 5, 6, 11]                  #p, e1, b1, x
            den_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'yatf_keys_k2tvpaebnwzixy1g1'
            arith = 1
            index = [3]     #p
            cols = [13, 8]  #g2, w8
            axis_slice = {}
            ffcfw_yatf = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights
                                     , weights=weights, na_weights=na_weights, den_weights=den_weights, keys=keys
                                     , arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_yatf = pd.concat([ffcfw_yatf],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_yatf = stacked_ffcfw_yatf.append(ffcfw_yatf)

        if report_run.loc['run_ffcfw_prog', 'Run']:
            type = 'stock'
            prod = 'ffcfw_prog_k3k5wzida0e0b0xyg2'
            na_prod = [2]
            prod_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight prod for propn of animals in e and b slice
            weights = 'prog_numbers_k3k5twzia0xg2'
            na_weights = [6,8,9,11] #d, e,b,y
            den_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight numbers for propn of animals in e and b slice
            keys = 'prog_keys_k3k5twzida0e0b0xyg2'
            arith = 1
            index = [3]             #w9
            cols = [1,4,10,11]  #k2, z, gender, g2
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            ffcfw_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod
                                                     , prod_weights=prod_weights, weights=weights
                                                     , na_weights=na_weights, den_weights=den_weights, keys=keys
                                                     , arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_prog = pd.concat([ffcfw_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_prog = stacked_ffcfw_prog.append(ffcfw_prog)

        if report_run.loc['run_ffcfw_offs', 'Run']:
            ##Average offs ffcfw with p, e & b axis. ffcfw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'ffcfw_offs_k3k5vpnw8zida0e0b0xyg3'
            na_prod = [2]
            prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = [4, 9, 11, 12]
            den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'offs_keys_k3k5tvpnwzidaebxyg3'
            arith = 1
            index = [9, 4]      #d, p. d here to save columns when many w
            cols = [15, 2, 6]   #g3, t, w
            axis_slice = {}
            axis_slice[11] = [0,1,1] #e: first cycle
            axis_slice[9] = [2,-1,1] #dam age: Adult
            # axis_slice[15] = [0,1,1] #g3: BBB
            ffcfw_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, prod_weights=prod_weights
                                     , weights=weights, den_weights=den_weights, na_prod=na_prod, na_weights=na_weights
                                     , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_offs = pd.concat([ffcfw_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_offs = stacked_ffcfw_offs.append(ffcfw_offs)

        if report_run.loc['run_nv_offs', 'Run']:
            ##Average offs NV with p, e & b axis. NV is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'nv_offs_k3k5vpnw8zida0e0b0xyg3'
            na_prod = [2]
            prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = [4,9,11,12]
            den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off hand)
            keys = 'offs_keys_k3k5tvpnwzidaebxyg3'
            arith = 1
            index =[4]
            cols =[0,13,1,2,6]  #k3,x,k5,t,w
            axis_slice = {}
            axis_slice[11] = [0,1,1] #first cycle
            axis_slice[9] = [2,-1,1] #Adult
            axis_slice[15] = [0,1,1] #BBB
            nv_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights
                                   , weights=weights, den_weights=den_weights, na_weights=na_weights
                                   , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            nv_offs = pd.concat([nv_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_nv_offs = stacked_nv_offs.append(nv_offs)

        if report_run.loc['run_lamb_survival', 'Run']:
            option = 0
            index =[2]
            cols =[5]
            axis_slice = {}
            lamb_survival = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols, axis_slice=axis_slice)
            lamb_survival = pd.concat([lamb_survival],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_lamb_survival = stacked_lamb_survival.append(lamb_survival)

        if report_run.loc['run_weanper', 'Run']:
            option = 1
            index =[2]
            cols =[]
            axis_slice = {}
            weanper = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols, axis_slice=axis_slice)
            weanper = pd.concat([weanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_weanper = stacked_weanper.append(weanper)

        if report_run.loc['run_scanper', 'Run']:
            option = 2
            index =[2]
            cols =[]
            axis_slice = {}
            scanper = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols, axis_slice=axis_slice)
            scanper = pd.concat([scanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_scanper = stacked_scanper.append(scanper)

        if report_run.loc['run_dry_propn', 'Run']:
            option = 3
            index =[2]
            cols =[]
            axis_slice = {}
            dry_propn = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols, axis_slice=axis_slice)
            dry_propn = pd.concat([dry_propn],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dry_propn = stacked_dry_propn.append(dry_propn)

        if report_run.loc['run_daily_mei_dams', 'Run']:
            type = 'stock'
            prod = 'mei_dams_k2p6ftva1nw8ziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [1, 2]
            den_weights = 'stock_days_k2p6ftva1nwziyg1'
            keys = 'dams_keys_k2p6ftvanwziy1g1'
            arith = 1               # for FP only
            index = [4, 1]          # [1]
            cols = [0, 2]           # [0]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_mei_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
            daily_mei_dams = pd.concat([daily_mei_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_mei_dams = stacked_daily_mei_dams.append(daily_mei_dams)

        if report_run.loc['run_daily_pi_dams', 'Run']:
            type = 'stock'
            prod = 'pi_dams_k2p6ftva1nw8ziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [1, 2]
            den_weights = 'stock_days_k2p6ftva1nwziyg1'
            keys = 'dams_keys_k2p6ftvanwziy1g1'
            arith = 1
            index =[4, 1]
            cols =[0, 2]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_pi_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                       na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                                       index=index, cols=cols, axis_slice=axis_slice)
            daily_pi_dams = pd.concat([daily_pi_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_pi_dams = stacked_daily_pi_dams.append(daily_pi_dams)

        if report_run.loc['run_daily_mei_offs', 'Run']:
            type = 'stock'
            prod = 'mei_offs_k3k5p6ftvnw8ziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = [2, 3]
            den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
            keys = 'offs_keys_k3k5p6ftvnwziaxyg3'
            arith = 1
            index =[5, 2]       #DVP
            cols =[13, 1, 3]    #g3, BTRT, nv
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_mei_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
            daily_mei_offs = pd.concat([daily_mei_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_mei_offs = stacked_daily_mei_offs.append(daily_mei_offs)

        if report_run.loc['run_daily_pi_offs', 'Run']:
            type = 'stock'
            prod = 'pi_offs_k3k5p6ftvnw8ziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = [2, 3]
            den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
            keys = 'offs_keys_k3k5p6ftvnwziaxyg3'
            arith = 1
            index =[5, 2]       #DVP
            cols =[13, 1, 3]    #g3, BTRT, nv
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_pi_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                       na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                                       index=index, cols=cols, axis_slice=axis_slice)
            daily_pi_offs = pd.concat([daily_pi_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_pi_offs = stacked_daily_pi_offs.append(daily_pi_offs)

        if report_run.loc['run_numbers_dams', 'Run']:
            type = 'stock'
            weights = 'dams_numbers_k2tvanwziy1g1'
            keys = 'dams_keys_k2tvanwziy1g1'
            arith = 2
            index =[2]
            cols =[0, 5]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_dams = pd.concat([numbers_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_dams = stacked_numbers_dams.append(numbers_dams)

        if report_run.loc['run_numbers_dams_p', 'Run']:
            type = 'stock'
            prod = 'on_hand_mort_k2tvpa1nwziyg1'
            weights = 'dams_numbers_k2tvanwziy1g1'
            na_weights = [3]
            keys = 'dams_keys_k2tvpanwziy1g1'
            arith = 2
            index =[2, 3]
            cols =[0, 1, 6]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_dams_p = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_dams_p = pd.concat([numbers_dams_p],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_dams_p = stacked_numbers_dams_p.append(numbers_dams_p)

        if report_run.loc['run_numbers_prog', 'Run']:
            type = 'stock'
            weights = 'prog_numbers_k3k5twzia0xg2'
            keys = 'prog_keys_k3k5twzia0xg2'
            arith = 2
            index =[2]
            cols =[0, 1, 5, 7]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_prog = pd.concat([numbers_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_prog = stacked_numbers_prog.append(numbers_prog)

        if report_run.loc['run_numbers_offs', 'Run']:
            type = 'stock'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            keys = 'offs_keys_k3k5tvnwziaxyg3'
            arith = 2
            index =[3]                  #DVP
            cols =[11, 9, 0, 1, 2, 5]   #g3, Gender, dam age, BTRT, t, w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_offs = pd.concat([numbers_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_offs = stacked_numbers_offs.append(numbers_offs)

        if report_run.loc['run_numbers_offs_p', 'Run']:
            type = 'stock'
            prod = 'on_hand_mort_k3k5tvpnwziaxyg3'
            weights = 'offs_numbers_k3k5tvnwziaxyg3'
            na_weights = [4]
            keys = 'offs_keys_k3k5tvpnwziaxyg3'
            arith = 2
            index =[4]              #p
            cols =[12, 0,10,1,2,6]  #dam age, gender, BTRT, t, w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_offs_p = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            numbers_offs_p = pd.concat([numbers_offs_p],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_offs_p = stacked_numbers_offs_p.append(numbers_offs_p)

        if report_run.loc['run_mort_dams', 'Run']:
            type = 'stock'
            prod = 'mort_pa1e1b1nwziyg1' #uses b axis instead of k for extra detail when scan=0
            weights = 1
            na_weights = []
            keys = 'dams_keys_paebnwziy1g1'
            arith = 4
            index =[0]          #period
            cols =[9, 3, 5]     #genotype, LSLN, w
            axis_slice = {}
            # axis_slice[0] = [2, 3, 1]
            # axis_slice[1] = [2, 4, 1]
            mort_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            mort_dams = pd.concat([mort_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_mort_dams = stacked_mort_dams.append(mort_dams)

        if report_run.loc['run_mort_offs', 'Run']:
            type = 'stock'
            prod = 'mort_pnwzida0e0b0xyg3' #uses b axis instead of k for extra detail when scan=0
            weights = 1
            na_weights = []
            keys = 'offs_keys_pnwzidaebxyg3'
            arith = 4
            index =[0]              #p
            cols =[11, 8, 2, 9]     #g3, BTRT, w, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            mort_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            mort_offs = pd.concat([mort_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_mort_offs = stacked_mort_offs.append(mort_offs)

        if report_run.loc['run_dse', 'Run']:
            ##you can go into f_dse to change the axis being reported.
            per_ha = True

            method = 0
            dse_sire, dse_dams, dse_offs = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
            dse_sire = pd.concat([dse_sire],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse_dams = pd.concat([dse_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse_offs = pd.concat([dse_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dse_sire = stacked_dse_sire.append(dse_sire)
            stacked_dse_dams = stacked_dse_dams.append(dse_dams)
            stacked_dse_offs = stacked_dse_offs.append(dse_offs)

            method = 1
            dse1_sire, dse1_dams, dse1_offs = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
            dse1_sire = pd.concat([dse1_sire],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse1_dams = pd.concat([dse1_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse1_offs = pd.concat([dse1_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dse1_sire = stacked_dse1_sire.append(dse1_sire)
            stacked_dse1_dams = stacked_dse1_dams.append(dse1_dams)
            stacked_dse1_offs = stacked_dse1_offs.append(dse1_offs)

        if report_run.loc['run_grnfoo', 'Run']:
            #returns foo at end of each FP
            type = 'pas'
            prod = 'foo_end_grnha_gop6lzt'
            na_prod = [0]
            weights = 'greenpas_ha_fgop6lzt'
            keys = 'keys_fgop6lzt'
            arith = 2
            index =[3]
            cols =[4]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            grnfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnfoo = pd.concat([grnfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grnfoo = stacked_grnfoo.append(grnfoo)

        if report_run.loc['run_dryfoo', 'Run']:
            #returns foo at end of each FP
            type = 'pas'
            prod = 1000
            weights = 'drypas_transfer_dp6zt'
            keys = 'keys_dp6zt'
            arith = 2
            index =[1]
            cols =[0]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            dryfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            dryfoo = pd.concat([dryfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dryfoo = stacked_dryfoo.append(dryfoo)

        if report_run.loc['run_napfoo', 'Run']:
            #returns foo at end of each FP
            prod = 1000
            type = 'pas'
            weights = 'nap_transfer_dp6zt'
            keys = 'keys_dp6zt'
            arith = 2
            index =[1]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            napfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            napfoo = pd.concat([napfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_napfoo = stacked_napfoo.append(napfoo)

        if report_run.loc['run_grncon', 'Run']:
            #returns consumption in each FP
            prod = 'cons_grnha_t_gop6lzt'
            type = 'pas'
            weights = 'greenpas_ha_fgop6lzt'
            keys = 'keys_fgop6lzt'
            arith = 2
            index =[3]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            grncon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grncon = pd.concat([grncon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grncon = stacked_grncon.append(grncon)

        if report_run.loc['run_drycon', 'Run']:
            #returns consumption in each FP
            prod = 1000
            type = 'pas'
            weights = 'drypas_consumed_fdp6zt'
            keys = 'keys_fdp6zt'
            arith = 2
            index =[2]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            drycon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drycon = pd.concat([drycon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drycon = stacked_drycon.append(drycon)

        if report_run.loc['run_grnnv', 'Run']:
            #returns NV during each FP regardless of whether selected or not
            type = 'pas'
            prod = 'nv_grnha_fgop6lzt'
            weights = 1
            keys = 'keys_fgop6lzt'
            arith = 5
            index = [3]
            cols = [2, 1]
            axis_slice = {}
            grnnv = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnnv = pd.concat([grnnv],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grnnv = stacked_grnnv.append(grnnv)

        if report_run.loc['run_grndmd', 'Run']:
            #returns DMD during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dmd_diet_grnha_gop6lzt'
            weights = 1
            keys = 'keys_gop6lzt'
            arith = 5
            index = [2]
            cols = [0, 1]
            axis_slice = {}
            grndmd = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grndmd = pd.concat([grndmd],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grndmd = stacked_grndmd.append(grndmd)

        if report_run.loc['run_avegrnfoo', 'Run']:
            #returns average FOO during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'foo_ave_grnha_gop6lzt'
            weights = 1
            keys = 'keys_gop6lzt'
            arith = 5
            index = [2]
            cols = [0, 1]
            axis_slice = {}
            grnfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnfoo = pd.concat([grnfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_avegrnfoo = stacked_avegrnfoo.append(grnfoo)

        if report_run.loc['run_drynv', 'Run']:
            #returns NV during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'nv_dry_fdp6zt'
            weights = 1
            keys = 'keys_fdp6zt'
            arith = 5
            index = [2]
            cols = [1]
            axis_slice = {}
            drynv = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drynv = pd.concat([drynv],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drynv = stacked_drynv.append(drynv)

        if report_run.loc['run_drydmd', 'Run']:
            #returns DMD during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dry_dmd_dp6zt'
            weights = 1
            keys = 'keys_dp6zt'
            arith = 5
            index = [1]
            cols = [0]
            axis_slice = {}
            drydmd = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drydmd = pd.concat([drydmd],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drydmd = stacked_drydmd.append(drydmd)

        if report_run.loc['run_dryfoo', 'Run']:
            #todo this report throws error when combining multiple excel... why??
            #returns average FOO during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dry_foo_dp6zt'
            weights = 1
            keys = 'keys_dp6zt'
            arith = 5
            index = [1]
            cols = [0]
            axis_slice = {}
            dryfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            dryfoo = pd.concat([dryfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_avedryfoo = stacked_avedryfoo.append(dryfoo)

        if report_run.loc['run_napcon', 'Run']:
            #returns consumption in each FP
            prod = 1000
            type = 'pas'
            weights = 'nap_consumed_fdp6zt'
            keys = 'keys_fdp6zt'
            arith = 2
            index =[2]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            napcon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            napcon = pd.concat([napcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_napcon = stacked_napcon.append(napcon)

        if report_run.loc['run_poccon', 'Run']:
            #returns consumption in each FP
            prod = 1000
            type = 'pas'
            weights = 'poc_consumed_fp6lz'
            keys = 'keys_fp6lz'
            arith = 2
            index =[1]
            cols =[3]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            poccon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            poccon = pd.concat([poccon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_poccon = stacked_poccon.append(poccon)

        if report_run.loc['run_supcon', 'Run']:
            #returns consumption in each FP
            option = 1
            supcon = rep.f_grain_sup_summary(lp_vars, r_vals, option=option)
            supcon = pd.concat([supcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_supcon = stacked_supcon.append(supcon)

        if report_run.loc['run_stubcon', 'Run']:
            #returns consumption in each FP
            stubcon = rep.f_stubble_summary(lp_vars, r_vals)
            stubcon = pd.concat([stubcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_stubcon = stacked_stubcon.append(stubcon)

    ####################################
    #run between trial reports and save#
    ####################################
    print("Writing to Excel")
    ##first check that excel is not open (microsoft puts a lock on files so they can't be updated from elsewhere while open)
    if os.path.isfile("Output/Report{0}.xlsx".format(processor)): #to check if report.xl exists
        while True:   # repeat until the try statement succeeds
            try:
                myfile = open("Output/Report{0}.xlsx".format(processor),"w") # chucks an error if excel file is open
                break                             # exit the loop
            except IOError:
                input("Could not open file! Please close Excel. Press Enter to retry.")
                # restart the loop

    ## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
    writer = pd.ExcelWriter('Output/Report{0}.xlsx'.format(processor),engine='xlsxwriter')

    ##make empty df to store row and col index settings. Used when combining multiple report.xl
    df_settings = pd.DataFrame(columns=['index', 'cols'])

    ##write to excel
    df_settings = rep.f_df2xl(writer, stacked_infeasible, 'infeasible', df_settings, option=1)
    df_settings = rep.f_df2xl(writer, stacked_non_exist,'Non-exist',df_settings,option=0,colstart=0)

    if report_run.loc['run_summary', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_summary, 'summary', df_settings, option=1)
    if report_run.loc['run_areasum', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_areasum, 'areasum', df_settings, option=1)
    if report_run.loc['run_pnl', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_pnl, 'pnl', df_settings, option=1)
    if report_run.loc['run_profitarea', 'Run']:
        plot = rep.f_xy_graph(stacked_profitarea)
        plot.savefig('Output/profitarea_curve.png')
    if report_run.loc['run_saleprice', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_saleprice, 'saleprice', df_settings, option=1)
    if report_run.loc['run_salevalue_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salevalue_offs, 'salevalue_offs', df_settings, option=1)
    if report_run.loc['run_salevalue_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salevalue_dams, 'salevalue_dams', df_settings, option=1)
    if report_run.loc['run_woolvalue_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_woolvalue_offs, 'woolvalue_offs', df_settings, option=1)
    if report_run.loc['run_woolvalue_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_woolvalue_dams, 'woolvalue_dams', df_settings, option=1)
    if report_run.loc['run_saledate_offs', 'Run']:
        stacked_saledate_offs = stacked_saledate_offs.astype(object)
        stacked_saledate_offs[stacked_saledate_offs==np.datetime64('1970-01-01')] = 0
        df_settings = rep.f_df2xl(writer, stacked_saledate_offs, 'saledate_offs', df_settings, option=1)
    if report_run.loc['run_cfw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_cfw_dams, 'cfw_dams', df_settings, option=1)
    if report_run.loc['run_fd_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_fd_dams, 'fd_dams', df_settings, option=1)
    if report_run.loc['run_cfw_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_cfw_offs, 'cfw_offs', df_settings, option=1)
    if report_run.loc['run_fd_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_fd_offs, 'fd_offs', df_settings, option=1)
    if report_run.loc['run_wbe_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_wbe_dams, 'wbe_dams', df_settings, option=1)
    if report_run.loc['run_wbe_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_wbe_offs, 'wbe_offs', df_settings, option=1)
    if report_run.loc['run_lw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_lw_dams, 'lw_dams', df_settings, option=1)
    if report_run.loc['run_ffcfw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_dams, 'ffcfw_dams', df_settings, option=1)
    if report_run.loc['run_ffcfw_yatf', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_yatf, 'ffcfw_yatf', df_settings, option=1)
    if report_run.loc['run_nv_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_nv_dams, 'nv_dams', df_settings, option=1)
    if report_run.loc['run_ffcfw_prog', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_prog, 'ffcfw_prog', df_settings, option=1)
    if report_run.loc['run_ffcfw_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_offs, 'ffcfw_offs', df_settings, option=1)
    if report_run.loc['run_nv_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_nv_offs, 'nv_offs', df_settings, option=1)
    if report_run.loc['run_lamb_survival', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_lamb_survival, 'lamb_survival', df_settings, option=1)
    if report_run.loc['run_weanper', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_weanper, 'wean_per', df_settings, option=1)
    if report_run.loc['run_scanper', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_scanper, 'scan_per', df_settings, option=1)
    if report_run.loc['run_dry_propn', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_dry_propn, 'dry_propn', df_settings, option=1)
    if report_run.loc['run_daily_mei_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_mei_dams, 'daily_mei_dams', df_settings, option=1)
    if report_run.loc['run_daily_pi_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_pi_dams, 'daily_pi_dams', df_settings, option=1)
    if report_run.loc['run_numbers_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_dams, 'numbers_dams', df_settings, option=1)
    if report_run.loc['run_numbers_dams_p', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_dams_p, 'numbers_dams_p', df_settings, option=1)
    if report_run.loc['run_numbers_prog', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_prog, 'numbers_prog', df_settings, option=1)
    if report_run.loc['run_numbers_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_offs, 'numbers_offs', df_settings, option=1)
    if report_run.loc['run_numbers_offs_p', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_offs_p, 'numbers_offs_p', df_settings, option=1)
    if report_run.loc['run_mort_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_mort_dams, 'mort_dams', df_settings, option=1)
    if report_run.loc['run_mort_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_mort_offs, 'mort_offs', df_settings, option=1)
    if report_run.loc['run_dse', 'Run']:
        dams_start_col = len(stacked_dse_sire.columns) + stacked_dse_sire.index.nlevels + 1
        offs_start_col = dams_start_col + len(stacked_dse_dams.columns) + stacked_dse_dams.index.nlevels + 1
        df_settings = rep.f_df2xl(writer, stacked_dse_sire, 'dse_wt', df_settings, option=0, colstart=0)
        df_settings = rep.f_df2xl(writer, stacked_dse_dams, 'dse_wt', df_settings, option=0, colstart=dams_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse_offs, 'dse_wt', df_settings, option=0, colstart=offs_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse1_sire, 'dse_mei', df_settings, option=0, colstart=0)
        df_settings = rep.f_df2xl(writer, stacked_dse1_dams, 'dse_mei', df_settings, option=0, colstart=dams_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse1_offs, 'dse_mei', df_settings, option=0, colstart=offs_start_col)
    if report_run.loc['run_grnfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grnfoo, 'grnfoo', df_settings, option=1)
    if report_run.loc['run_dryfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_dryfoo, 'dryfoo', df_settings, option=1)
    if report_run.loc['run_napfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_napfoo, 'napfoo', df_settings, option=1)
    if report_run.loc['run_grncon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grncon, 'grncon', df_settings, option=1)
    if report_run.loc['run_drycon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drycon, 'drycon', df_settings, option=1)
    if report_run.loc['run_grnnv', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grnnv, 'grnnv', df_settings, option=1)
    if report_run.loc['run_grndmd', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grndmd, 'grndmd', df_settings, option=1)
    if report_run.loc['run_avegrnfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_avegrnfoo, 'avegrnfoo', df_settings, option=1)
    if report_run.loc['run_drynv', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drynv, 'drynv', df_settings, option=1)
    if report_run.loc['run_drydmd', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drydmd, 'drydmd', df_settings, option=1)
    if report_run.loc['run_avedryfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_avedryfoo, 'avedryfoo', df_settings, option=1)
    if report_run.loc['run_napcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_napcon, 'napcon', df_settings, option=1)
    if report_run.loc['run_poccon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_poccon, 'poccon', df_settings, option=1)
    if report_run.loc['run_supcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_supcon, 'supcon', df_settings, option=1)
    if report_run.loc['run_stubcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_stubcon, 'stubcon', df_settings, option=1)


    df_settings.to_excel(writer, 'df_settings')
    writer.save()

    print("Report complete. Processor: {0}".format(processor))


if __name__ == '__main__':
    ##read in exp log
    exp_data, experiment_trials = fun.f_read_exp()

    ##check if trial results are up to date. Out-dated if:
    ##  1. exp.xls has changed
    ##  2. any python module has been updated
    ##  3. the trial needed to be run last time but the user opted not to run that trial
    exp_data = fun.f_run_required(exp_data)
    exp_data = fun.f_group_exp(exp_data, experiment_trials)  # cut exp_data based on the experiment group
    trial_outdated = exp_data['run_req']  # returns true if trial is out of date

    ## enter the trials to summarise and the reports to include
    trials = np.array(exp_data.index.get_level_values(3))[
        pd.Series(exp_data.index.get_level_values(2)).fillna(0).astype(
            bool)]  # this is slightly complicated because blank rows in exp.xl result in nan, so nan must be converted to 0.

    ##check the trials you want to run exist and are up to date - if trial doesnt exist it is removed from trials to
    # report array so that the others can still be run. A list of trials that don't exist is the 'non_exist' sheet in report excel.
    trials, non_exist_trials = rep.f_errors(trial_outdated,trials)

    ##clear the old report.xlsx
    for f in glob.glob("Output/Report*.xlsx"):
        os.remove(f)


    ##print out the reports being run and number of trials
    print('Number of trials to run: ', len(trials))
    print("The following reports will be run: \n", report_run.index[report_run.loc[:,'Run']])

    ##determine the processor for each report
    ## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
    try:
        maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
    except IndexError:  # in case no arg passed to python
        maximum_processes = 1

    ##start multiprocessing
    ### number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
    agents = min(multiprocessing.cpu_count(), len(trials), maximum_processes)
    ###set up dataset for f_report
    args = []
    for agent in list(range(agents)):
        start_trial = int(len(trials)/agents * agent)
        end_trial = int(len(trials)/agents * (agent+1))
        process_trials = trials[start_trial:end_trial]
        arg = [agent,process_trials, non_exist_trials]
        args.append(arg)
    with multiprocessing.Pool(processes=agents) as pool:
        pool.starmap(f_report, args)

    end = time.time()
    # print("Reports successfully completed")
    print(f'Reporting successfully completed at: {time.ctime()}, total time taken: {end - start:.2f}')
