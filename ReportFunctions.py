"""

This module contains function which are used for three main jobs:

#.  General jobs including error checking and reading and writing output files.
#.  'within trial' calculations. These functions manipulate the output variables from the LP and the
    values from the precalcs. These functions all return a data array.
#.  'between trial' calculations. These functions summarise the data array from the 'within trial' functions
    for all the trials in an experiment. These functions can return table or figures.

.. note:: This module should not import inputs (in case the inputs are adjusted during the exp
    so they will not be correct for r_vals)

.. tip:: There are cases where the report requires more detail than the decision variable returned from
    the lp contains. For example some livestock reports (e.g. NV) need an e b and p axis but the
    clustering has removed this level of detail in the lp variables.
    To handle this we make the corresponding r_vals array with both the detailed and clustered axis.

author: young
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import os.path
import sys
import xlsxwriter

import Functions as fun
import Exceptions as exc

na = np.newaxis

###################
#general functions#
###################
def f_df2xl(writer, df, sheet, df_settings=None, rowstart=0, colstart=0, option=0):
    '''
    Pandas to excel. https://xlsxwriter.readthedocs.io/working_with_pandas.html

        - You can simply stick a dataframe from pandas into excel using df.to_excel() function.
          for this you can specify the workbook the sheet and the start row or col (so you can put
          multiple dfs in one sheet)
        - The next level involves interacting with xlsxwriter. This allows you to do custom things like
          creating graphs, hiding rows/cols, filtering or grouping.

    :param writer: writer used. controls the workbook being writen to.
    :param df: dataframe going to excel
    :param sheet: str: sheet name.
    :param df_settings: df: df to store number of row and col indexes.
    :param rowstart: start row in excel
    :param colstart: start col in excel
    :param option: int: specifying the writing option
                    0: df straight into excel
                    1: df into excel collapsing empty rows and cols
    '''
    ##store df settings
    if df_settings is not None:
        df_settings.loc[sheet] = [df.index.nlevels, df.columns.nlevels]

    ## simple write df to xl
    df.to_excel(writer, sheet, startrow=rowstart, startcol=colstart)

    ##set up xlsxwriter stuff needed for advanced options
    workbook = writer.book
    worksheet = writer.sheets[sheet]

    ## collapse rows and cols with all 0's
    if option==1:
        df = df.round(5)  # round so that very small numbers are dropped out in the next step
        for row in range(len(df)-1):   #todo: in range(len(df)) hides the last blank row but causes a blank line in some of report.xl
            if (df.iloc[row]==0).all():
                offset = df.columns.nlevels #number of columns used for names
                if offset>1:
                    offset += 1 #for some reason if the cols are multiindex the an extra row gets added when writing to excel
                worksheet.set_row(row+offset,None,None,{'level': 1, 'hidden': True}) #set hidden to true to collapse the level initially

        for col in range(len(df.columns)):
            if (df.iloc[:,col]==0).all():
                offset = df.index.nlevels
                col = xlsxwriter.utility.xl_col_to_name(col+offset) + ':' + xlsxwriter.utility.xl_col_to_name(col+offset) #convert col number to excel col reference eg 'A:B'
                worksheet.set_column(col,None,None,{'level': 1, 'hidden': True})


    ##apply filter
    if option==2:
        # Activate autofilter
        worksheet.autofilter(f'B1:B{len(df)}')
        worksheet.filter_column('B', 'x < 5') # todo this will need to become function argument

        # Hide the rows that don't match the filter criteria.
        for idx,row_data in df.iterrows():
            region = row_data['Data']
            if not (region < 5):
                # We need to hide rows that don't match the filter.
                worksheet.set_row(idx + 1,options={'hidden': True})

    ##create chart
    if option==3:
        # Create a chart object.
        chart = workbook.add_chart({'type': 'column'}) # todo this will need to become function argument
        # Configure the series of the chart from the dataframe data.
        chart.add_series({'values': '=areasum!$B$2:$B$8'}) # todo this will need to become function argument
        # Insert the chart into the worksheet.
        worksheet.insert_chart('D2',chart)

    return df_settings

def f_errors(trial_outdated, trials):
    '''
    The report module conducts three error checks before commencing:

    #. Have you run the main model for each trial you are trying to report. If not a warning will be printed and
       the given trial will be removed from the list of trials to report (allowing the remaining trials to still
       be reported).
    #. Are any trials out of date E.g. have you run the main model since updating the inputs or the code.
       If trials are out of date a warning message will be printed but the report code will continue to execute.
    #. Did all the trials you are reporting solve optimally. Infeasible trials are still reported the reports are just
       filled with 0's. A list of infeasible trials will be printed to the console and reported in the final Report.xlsx
       document.

    :param trial_outdated: boolean list of all trials stating if trial is out of date.
    :param trials: list of trials being run
    :return: None.
    '''
    ##first check if data exists for each desired trial
    infeasible_trials=[]
    for trial_name in trials:
        if os.path.isfile('Output/infeasible/{0}.txt'.format(trial_name)):
            infeasible_trials.append(trial_name)
        else:
            pass
    if infeasible_trials:
        print("Infeasible trials being reported:\n", infeasible_trials)

    ##second check if data exists for each desired trial
    non_exist_trials = []
    for trial_name in trials:
        if os.path.isfile('pkl/pkl_r_vals_{0}.pkl'.format(trial_name)):
            pass
        else:
            print('''WARNING: Trials for reporting don't all exist''')
            trials = trials[trials!=trial_name] #remove trials that don't exist from the list of trials to run.
            non_exist_trials.append(trial_name)

    ##third check if generating results using out of date data.
    outdatedbool = trial_outdated.loc[(slice(None), slice(None), slice(None), trials)].values  # have to use the trial name because the order is different
    if any(outdatedbool):  # have to use the trial name because the order is different
        print('''

              Generating reports from out dated data: Trial %s
                
              ''' %np.array(trials)[outdatedbool])
    return trials, non_exist_trials

def load_pkl(trial_name):
    '''load in lp_vars and r_vals output file.
    '''
    with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"rb") as f:
        lp_vars = pkl.load(f)
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"rb") as f:
        r_vals = pkl.load(f)
    return lp_vars, r_vals

def f_vars2np(lp_vars, var_key, shape, keys_z, z_pos):
    '''
    converts lp_vars to numpy.
    :param lp_vars: dict of lp variables
    :param var_key: string - name of variable to convert to numpy
    :param shape: shape of desired numpy array
    :param z_pos: position to add z axis
    :return: numpy array with season axis.
    '''
    final_vars = np.zeros(shape)
    if isinstance(shape,int):
        shape_wo_z = 1
        len_z = shape
        len_shape = 1
    elif z_pos == -1:
        shape_wo_z = shape[0:z_pos] + (1,) #make z singleton
        len_z = shape[z_pos]
        len_shape = len(shape)
    else:
        shape_wo_z = shape[0:z_pos] + (1,)+ shape[z_pos+1:] #make z singleton
        len_z = shape[z_pos]
        len_shape = len(shape)
    for z in range(len_z):
        z_key = keys_z[z]
        try:
            vars = np.array(list(lp_vars[z_key][var_key].values()))
        except KeyError:
            vars = np.array(list(lp_vars.values()))
        vars = vars.reshape(shape_wo_z)
        vars[vars == None] = 0  # replace None with 0
        slc = [slice(None)] * len_shape
        slc[z_pos] = slice(z,z+1)
        final_vars[tuple(slc)] = vars
    return final_vars

def f_vars2df(lp_vars, var_key, z_keys):
    '''
    converts lp_vars to pandas series.
    :param lp_vars: dict of variables.
    :param var_key: string - name of variable to convert to series.
    :return: series with season as index level 0
    '''
    for z_key, z in zip(z_keys,range(len(z_keys))):
        var_series = pd.Series(lp_vars[z_key][var_key])
        var_series = pd.concat([var_series], keys=[z_key])
        if z == 0:
            final_series = var_series
        else:
            final_series = pd.concat([final_series, var_series])
    return final_series.sort_index()


########################
# across trial reports #
########################
def f_xy_graph(data):
    '''Generic x-y line graphing function.

    :param data: df with data to plot. First col contains x values and second col contains y values
    :return: x-y plot

    '''
    ##loop through trials and generate pnl table
    x_vals = data.iloc[:,0]  # create list to append pnl table from each trial
    y_vals = data.iloc[:,1]  # create list to append pnl table from each trial
    plt.plot(x_vals, y_vals)
    return plt


###################
# input summaries #
###################

def f_price_summary(lp_vars, r_vals, option, grid, weight, fs):
    '''Returns price summaries

    :param r_vals:
    :param option:

            #. farmgate grain price
            #. wool price STB price for FNF (free or nearly free of fault)
            #. sale price for specified grid at given weight and fat score

    :param grid: list - sale grids to report. Has to be int between 0 and 7 inclusive.
    :param weight: float/int - stock weight to report price for.
    :param fs: int - fat score to report price for. Has to be number between 1-5 inclusive.
    :return: price summary df
    '''

    ##grain price - farmgate (price received by farmer)
    if option == 0:
        return r_vals['crop']['farmgate_price']

    ##wool price - grid price
    if option == 1:
        return pd.Series(r_vals['stock']['woolp_mpg_w4'], index=r_vals['fd_range'])

    ##sale price - grid price
    if option == 2:
        ###create dataframe
        sale_index = pd.MultiIndex(levels=[[], [], []],
                                   codes=[[], [], []],
                                   names=['Grid', 'Weight', 'Fat Score'])
        saleprice = pd.DataFrame(index=sale_index, columns=['Price $/kg',
                                                            'Price $/hd'])  # need to initialise df with multiindex so rows can be added

        grid_price_s7s5s6 = r_vals['stock']['grid_price_s7s5s6']
        weight_range_s7s5 = r_vals['stock']['weight_range_s7s5']
        grid_keys = r_vals['stock']['salegrid_keys']
        for t_grid, t_weight, t_fs in zip(grid, weight, fs):
            ##grid name - used in table index
            grid_name = grid_keys[t_grid]
            ##index grid and fs
            price_s5 = grid_price_s7s5s6[t_grid, :, t_fs]
            ##interpolate to get price for specified weight
            lookup_weights = weight_range_s7s5[t_grid, :]
            price = np.interp(t_weight, lookup_weights, price_s5)
            ##attach to df
            ###if price is less than 10 it is assumed to be $/kg else $/hd
            if price < 10:
                col = 'Price $/kg'
            else:
                col = 'Price $/hd'
            saleprice.loc[(grid_name, t_weight, t_fs), col] = price
        return saleprice


#########################################
# intermediate report building functions#
#########################################

def f_summary(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['obj', 'profit', 'SR', 'Pas %', 'Sup'])
    ##obj
    summary_df.loc[trial, 'obj'] = f_profit(lp_vars, r_vals, option=0)
    ##profit - no minroe and asset
    summary_df.loc[trial, 'profit'] = f_profit(lp_vars, r_vals, option=1)
    ##total dse/ha in fp0
    summary_df.loc[trial, 'SR'] = f_dse(lp_vars, r_vals, method=0, per_ha=True, summary=True)
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)
    ##supplement
    summary_df.loc[trial, 'Sup'] = f_grain_sup_summary(lp_vars,r_vals,option=3)
    return summary_df




def f_rotation(lp_vars, r_vals):
    '''
    manipulates the rotation solution into usable format. This is used in many function.
    '''
    ##rotation
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(5, append=True)  # add landuse as index level
    keys_z = r_vals['stock']['keys_z']
    rot_area_zrl = f_vars2df(lp_vars, 'v_phase_area', keys_z) # create a series of all the phase areas, need to sort the index because it was chuck error for some calculations
    rot_area_zlrk = rot_area_zrl.unstack(1).reindex(phases_rk.index, axis=1, level=0).stack([0,1])  # add landuse to the axis
    return phases_rk, rot_area_zrl, rot_area_zlrk


def f_area_summary(lp_vars, r_vals, option):
    '''
    Rotation & landuse area summary. With multiple output levels.
    return options:

    :param lp_vars: dict
    :param r_vals: dict
    :key option:

        #. tuple all results wrapped in tuple
        #. table all rotations by lmu
        #. float total pasture area
        #. float total crop area
        #. table crop and pasture area by lmu
        #. float pasture %

    '''

    ##read from other functions
    rot_area_zrl, rot_area_zlrk = f_rotation(lp_vars, r_vals)[1:3]
    landuse_area_k_zl = rot_area_zlrk.sum(axis=0, level=(0,1,3)).unstack([0,1])  # area of each landuse (sum lmu and rotation)

    ##all rotations by lmu
    rot_area_zr_l = rot_area_zrl.unstack()
    if option == 1:
        return rot_area_zr_l.round(0)

    ###crop & pasture area
    ####you can now use isin pasture or crop sets to calc the area of crop or pasture
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    pasture_area_l = landuse_area_k_zl[landuse_area_k_zl.index.isin(all_pas)].sum()  # sum landuse
    if option == 2:
        return pasture_area_l.sum().round(0)
    crop_area_l = landuse_area_k_zl[~landuse_area_k_zl.index.isin(all_pas)].sum()  # sum landuse
    if option == 3:
        return crop_area_l.sum().round(0)

    ##crop & pasture area by lmu
    croppas_area_l = pd.DataFrame()
    croppas_area_l['pasture'] = pasture_area_l
    croppas_area_l['crop'] = crop_area_l
    if option == 4:
        return croppas_area_l.round(0)

    if option==5:
        return (fun.f_divide(pasture_area_l.sum(), rot_area_zr_l.to_numpy().sum()) * 100).round(1)

    ##return all if option==0
    if option == 0:
        return rot_area_zr_l, pasture_area_l, crop_area_l, croppas_area_l


def f_mach_summary(lp_vars, r_vals, option=0):
    '''
    Machine summary.
    :param option:

        #. table: total machine cost for each crop in each cash period

    '''
    ##call rotation function to get rotation info
    phases_rk, rot_area_zrl = f_rotation(lp_vars, r_vals)[0:2]
    keys_z = r_vals['stock']['keys_z']

    ##harv
    contractharv_hours_zk = f_vars2df(lp_vars, 'v_contractharv_hours', keys_z)
    harv_hours_zk = f_vars2df(lp_vars, 'v_harv_hours', keys_z).sum(level=(0,2))  # sum p5 axis
    contract_harvest_cost_c_zk = r_vals['mach']['contract_harvest_cost'].T.T.swaplevel(0,1,axis=1).sort_index(axis=1)
    own_harvest_cost_c_zk = r_vals['mach']['harvest_cost'].T.T.swaplevel(0,1,axis=1).sort_index(axis=1)
    harvest_cost_c_zk = contract_harvest_cost_c_zk.mul(contractharv_hours_zk, axis=1) + own_harvest_cost_c_zk.mul(harv_hours_zk, axis=1)

    ##seeding
    seeding_days_kl_z = f_vars2df(lp_vars, 'v_seeding_machdays', keys_z).sum(level=(0, 2,3)).unstack(0)  # sum labour period axis
    seeding_rate_kl = r_vals['mach']['seeding_rate'].stack()
    seeding_ha_kl_z = seeding_days_kl_z.mul(seeding_rate_kl.reindex(seeding_days_kl_z.index), axis=0) # note seeding ha won't equal the rotation area because arable area is included in seed_ha.
    seeding_cost_cz_l = r_vals['mach']['seeding_cost'].stack()
    seeding_cost_c_klz = seeding_cost_cz_l.reindex(seeding_ha_kl_z.index, axis=1, level=1).unstack()
    seeding_cost_own_c_zk = seeding_cost_c_klz.mul(seeding_ha_kl_z.stack(), axis=1).sum(axis=1, level=(0,2)).swaplevel(0,1,axis=1)  # sum lmu axis
    contractseeding_ha_zk = f_vars2df(lp_vars, 'v_contractseeding_ha', keys_z).sum(level=(0,2))  # sum labour period and lmu axis
    contractseed_cost_ha_c_z = r_vals['mach']['contractseed_cost']
    contractseed_cost_ha_c_zk = contractseed_cost_ha_c_z.reindex(contractseeding_ha_zk.index, axis=1, level=0)
    seeding_cost_contract_c_zk =  contractseed_cost_ha_c_zk.mul(contractseeding_ha_zk, axis=1, level=1)

    ##fert & chem mach cost
    fert_app_cost_rzl_c = r_vals['crop']['fert_app_cost']
    nap_fert_app_cost_rzl_c = r_vals['crop']['nap_fert_app_cost'].unstack().reindex(fert_app_cost_rzl_c.unstack().index, axis=0,level=0).stack()
    chem_app_cost_ha_rzl_c = r_vals['crop']['chem_app_cost_ha']
    fertchem_cost_rzl_c = pd.concat([fert_app_cost_rzl_c, nap_fert_app_cost_rzl_c, chem_app_cost_ha_rzl_c], axis=1).sum(axis=1, level=0)  # cost per ha
    fertchem_cost_rz_c = fertchem_cost_rzl_c.mul(rot_area_zrl.swaplevel(0,1).sort_index(), axis=0).sum(axis=0, level=(0,1))  # mul area and sum lmu
    fertchem_cost_c_zk = fertchem_cost_rz_c.unstack(1).reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                                      level=1).stack(0).unstack(0)  # reindex to include landuse and sum rot

    ##insurance
    mach_insurance_c = r_vals['mach']['mach_insurance']

    ##conbime all costs
    exp_mach_c_zk = pd.concat([fertchem_cost_c_zk, seeding_cost_own_c_zk, seeding_cost_contract_c_zk, harvest_cost_c_zk
                               ], axis=0).sum(axis=0, level=0)
    ##return all if option==0
    if option == 0:
        return exp_mach_c_zk, mach_insurance_c


def f_grain_sup_summary(lp_vars, r_vals, option=0):
    '''
    Summary of grain, supplement and their costs

    :param option: int:

            #. return dict with various elements
            #. return total supplement fed in each feed period
            #. return total of each grain supplement fed in each feed period
            #. return total sup fed

    '''
    ##create dict to store grain variables
    grain = {}
    keys_z = r_vals['stock']['keys_z']
    ##prices
    grains_sale_price_kg_c = r_vals['crop']['grain_price'].T.stack()
    grains_buy_price_kg_c = r_vals['sup']['buy_grain_price'].T.stack()

    ##grain purchased
    grain_purchased_zkg = f_vars2df(lp_vars, 'v_buy_grain', keys_z)

    ##grain sold
    grain_sold_zkg = f_vars2df(lp_vars, 'v_sell_grain', keys_z)

    ##grain fed
    grain_fed_zkgvp6 = f_vars2df(lp_vars, 'v_sup_con', keys_z)
    grain_fed_zkg = grain_fed_zkgvp6.sum(level=(0, 1, 2))  # sum feed pool and feed period
    grain_fed_zkp6 = grain_fed_zkgvp6.sum(level=(0, 1, 4))  # sum feed pool and grain pool
    grain_fed_zp6 = grain_fed_zkgvp6.sum(level=(0, 4))  # sum feed pool, landuse and grain pool
    if option == 1:
        return grain_fed_zp6.to_frame()
    if option == 2:
        return grain_fed_zkp6.unstack(0)
    if option == 3:
        return grain_fed_zkgvp6.sum().round(1)
    ##total grain produced by crop enterprise
    total_grain_produced_zkg = grain_sold_zkg + grain_fed_zkg - grain_purchased_zkg  # total grain produced by crop enterprise
    grains_sale_price_zkg_c = grains_sale_price_kg_c.unstack().reindex(total_grain_produced_zkg.unstack().index, axis=0,level=1).stack()
    rev_grain_k_cz = grains_sale_price_zkg_c.mul(total_grain_produced_zkg, axis=0).unstack(0).sum(axis=0, level=0)  # sum grain pool, have to reindex (not really sure why since it is the same index - maybe one has been condensed ie index with nan removed)
    grain['rev_grain_k_cz'] = rev_grain_k_cz

    ##supplementary cost: cost = sale_price * (grain_fed - grain_purchased) + buy_price * grain_purchased
    grains_buy_price_zkg_c = grains_buy_price_kg_c.unstack().reindex(grain_purchased_zkg.unstack().index, axis=0,level=1).stack()
    sup_exp_z_c = (grains_sale_price_zkg_c.mul(grain_fed_zkg - grain_purchased_zkg, axis=0)
                 + grains_buy_price_zkg_c.mul(grain_purchased_zkg, axis=0)).sum(axis=0,level=0)  # sum grain pool
    grain['sup_exp_c_z'] = sup_exp_z_c.T
    return grain


def f_stubble_summary(lp_vars, r_vals):
    keys_z = r_vals['stock']['keys_z']
    stub_zfp6ks = f_vars2df(lp_vars, 'v_stub_con', keys_z)
    return stub_zfp6ks.sum(level=(0,2, 4)).unstack()


def f_crop_summary(lp_vars, r_vals, option=0):
    '''
    Crop summary. Includes pasture inputs.
    :param option:

        #. Return - tuple: fert cost, chem cost, miscellaneous costs and grain revenue for each landuse

    '''
    ##call rotation function to get rotation info
    phases_rk, rot_area_zrl = f_rotation(lp_vars, r_vals)[0:2]
    keys_z = r_vals['stock']['keys_z']
    ##expenses
    ###fert
    nap_phase_fert_cost_r_cl = r_vals['crop']['nap_phase_fert_cost'].unstack()
    index = pd.MultiIndex.from_product([keys_z, nap_phase_fert_cost_r_cl.index])
    nap_phase_fert_cost_zrl_c = nap_phase_fert_cost_r_cl.reindex(index, axis=0,level=0).stack()
    phase_fert_cost_zrl_c = r_vals['crop']['phase_fert_cost'].swaplevel(0,1,axis=0)
    exp_fert_ha_zrl_c = pd.concat([phase_fert_cost_zrl_c, nap_phase_fert_cost_zrl_c], axis=1).sum(axis=1, level=0)
    exp_fert_zr_c = exp_fert_ha_zrl_c.mul(rot_area_zrl, axis=0).sum(axis=0, level=(0,1))  # mul area and sum lmu
    exp_fert_k_cz = exp_fert_zr_c.unstack(0).reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ###chem
    chem_cost_zrl_c = r_vals['crop']['chem_cost'].swaplevel(0,1,axis=0).sort_index()
    exp_chem_zr_c = chem_cost_zrl_c.mul(rot_area_zrl, axis=0).sum(axis=0, level=(0,1))  # mul area and sum lmu
    exp_chem_k_cz = exp_chem_zr_c.unstack(0).reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ###misc
    stub_cost_zrl_c = r_vals['crop']['stub_cost'].swaplevel(0,1,axis=0)
    insurance_cost_zrl_c = r_vals['crop']['insurance_cost'].swaplevel(0,1,axis=0)
    seedcost_zrl_c = r_vals['crop']['seedcost'].swaplevel(0,1,axis=0)
    misc_exp_ha_zrl_c = pd.concat([stub_cost_zrl_c, insurance_cost_zrl_c, seedcost_zrl_c], axis=1).sum(axis=1, level=0)  # stubble, seed & insurance
    misc_exp_zr_c = misc_exp_ha_zrl_c.reindex(rot_area_zrl.index, axis=0).mul(rot_area_zrl, axis=0).sum(axis=0, level=(0,1))  # mul area and sum lmu
    misc_exp_k_cz = misc_exp_zr_c.unstack(0).reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ##revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    rev_grain_k_cz = grain_summary['rev_grain_k_cz']
    ##return all if option==0
    if option == 0:
        return exp_fert_k_cz, exp_chem_k_cz, misc_exp_k_cz, rev_grain_k_cz


def f_stock_reshape(lp_vars, r_vals):
    '''
    Stock reshape. Gets everything into the correct shape.
    Returns a dictionary with stock params.
    '''
    ##keys
    keys_c = r_vals['fin']['keys_c']
    keys_a = r_vals['stock']['keys_a']
    keys_d = r_vals['stock']['keys_d']
    keys_g0 = r_vals['stock']['keys_g0']
    keys_g1 = r_vals['stock']['keys_g1']
    keys_g2 = r_vals['stock']['keys_g2']
    keys_g3 = r_vals['stock']['keys_g3']
    keys_f = r_vals['stock']['keys_f']
    keys_h1 = r_vals['stock']['keys_h1']
    keys_i = r_vals['stock']['keys_i']
    keys_k2 = r_vals['stock']['keys_k2']
    keys_k3 = r_vals['stock']['keys_k3']
    keys_k5 = r_vals['stock']['keys_k5']
    keys_lw1 = r_vals['stock']['keys_lw1']
    keys_lw3 = r_vals['stock']['keys_lw3']
    keys_lw_prog = r_vals['stock']['keys_lw_prog']
    keys_n1 = r_vals['stock']['keys_n1']
    keys_n3 = r_vals['stock']['keys_n3']
    keys_p8 = r_vals['stock']['keys_p8']
    keys_t1 = r_vals['stock']['keys_t1']
    keys_t2 = r_vals['stock']['keys_t2']
    keys_t3 = r_vals['stock']['keys_t3']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']
    keys_y0 = r_vals['stock']['keys_y0']
    keys_y1 = r_vals['stock']['keys_y1']
    keys_y3 = r_vals['stock']['keys_y3']
    keys_x = r_vals['stock']['keys_x']
    keys_z = r_vals['stock']['keys_z']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_pastures = r_vals['pas']['keys_pastures']

    ##axis len
    len_a = len(keys_a)
    len_c = len(keys_c)
    len_d = len(keys_d)
    len_g0 = len(keys_g0)
    len_g1 = len(keys_g1)
    len_g2 = len(keys_g2)
    len_g3 = len(keys_g3)
    len_f = len(keys_f)
    len_h1 = len(keys_h1)
    len_i = len(keys_i)
    len_k2 = len(keys_k2)
    len_k3 = len(keys_k3)
    len_k5 = len(keys_k5)
    len_lw1 = len(keys_lw1)
    len_lw3 = len(keys_lw3)
    len_lw_prog = len(keys_lw_prog)
    len_n1 = len(keys_n1)
    len_n3 = len(keys_n3)
    len_p8 = len(keys_p8)
    len_t1 = len(keys_t1)
    len_t2 = len(keys_t2)
    len_t3 = len(keys_t3)
    len_v1 = len(keys_v1)
    len_v3 = len(keys_v3)
    len_y0 = len(keys_y0)
    len_y1 = len(keys_y1)
    len_y3 = len(keys_y3)
    len_x = len(keys_x)
    len_z = len(keys_z)
    len_p6 = len(keys_p6)
    len_p5 = len(keys_p5)

    ##create dict for reshaped variables
    stock_vars = {}


    ##animal numbers
    ###shapes
    sire_shape = len_z, len_g0
    dams_shape = len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    prog_shape = len_k3, len_k5, len_t2, len_lw_prog, len_z, len_i, len_a, len_x, len_g2
    offs_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    infra_shape = len_h1, len_z
    ###sire
    stock_vars['sire_numbers_zg0'] = f_vars2np(lp_vars, 'v_sire', sire_shape, keys_z, z_pos=-2).astype(float)
    ###dams
    stock_vars['dams_numbers_k2tvanwziy1g1'] = f_vars2np(lp_vars, 'v_dams', dams_shape, keys_z, z_pos=-4).astype(float)
    ###prog
    stock_vars['prog_numbers_k3k5twzia0xg2'] = f_vars2np(lp_vars, 'v_prog', prog_shape, keys_z, z_pos=-5).astype(float)
    ###offs
    stock_vars['offs_numbers_k3k5tvnwziaxyg3'] = f_vars2np(lp_vars, 'v_offs', offs_shape, keys_z, z_pos=-6).astype(float)
    ###infrastructure
    stock_vars['infrastructure_h1z'] = f_vars2np(lp_vars, 'v_infrastructure', infra_shape, keys_z, z_pos=-1).astype(float)

    return stock_vars


def f_pasture_reshape(lp_vars, r_vals):
    '''
    Reshape pasture lp variables into numpy array

    :param lp_vars: lp variables
    :return: dict
    '''
    keys_d = r_vals['pas']['keys_d']
    keys_f = r_vals['pas']['keys_f']
    keys_p6 = r_vals['pas']['keys_p6']
    keys_g = r_vals['pas']['keys_g']
    keys_k = r_vals['pas']['keys_k']
    keys_l = r_vals['pas']['keys_l']
    keys_o = r_vals['pas']['keys_o']
    keys_p5 = r_vals['pas']['keys_p5']
    keys_r = r_vals['pas']['keys_r']
    keys_t = r_vals['pas']['keys_t']
    keys_z = r_vals['stock']['keys_z']

    len_d = len(keys_d)
    len_f = len(keys_f)
    len_p6 = len(keys_p6)
    len_g = len(keys_g)
    len_k = len(keys_k)
    len_l = len(keys_l)
    len_o = len(keys_o)
    len_p5 = len(keys_p5)
    len_r = len(keys_r)
    len_t = len(keys_t)
    len_z = len(keys_z)

    ##dict to store reshaped pasture stuff in
    pas_vars = {}

    # store keys - must be in axis order
    pas_vars['keys_fgop6lzt'] = [keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    pas_vars['keys_gop6lzt'] = [keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    pas_vars['keys_fdp6zt'] = [keys_f, keys_d, keys_p6, keys_z, keys_t]
    pas_vars['keys_dp6zt'] = [keys_d, keys_p6, keys_z, keys_t]
    pas_vars['keys_dp6zt'] = [keys_d, keys_p6, keys_z, keys_t]
    pas_vars['keys_fp6lz'] = [keys_f, keys_p6, keys_l, keys_z]

    ##shapes
    fgop6lzt = len_f, len_g, len_o, len_p6, len_l, len_z, len_t
    fdp6zt = len_f, len_d, len_p6, len_z, len_t
    dp6zt = len_d, len_p6, len_z, len_t
    fp6lz = len_f, len_p6, len_l, len_z

    ##reshape green pasture hectare variable
    pas_vars['greenpas_ha_fgop6lzt'] = f_vars2np(lp_vars, 'v_greenpas_ha', fgop6lzt, keys_z, z_pos=-2)

    ##dry end period
    pas_vars['drypas_transfer_dp6zt'] = f_vars2np(lp_vars, 'v_drypas_transfer', dp6zt, keys_z, z_pos=-2)

    ##nap end period
    pas_vars['nap_transfer_dp6zt'] = f_vars2np(lp_vars, 'v_nap_transfer', dp6zt, keys_z, z_pos=-2)

    ##dry consumed
    pas_vars['drypas_consumed_fdp6zt'] = f_vars2np(lp_vars, 'v_drypas_consumed', fdp6zt, keys_z, z_pos=-2)

    ##nap consumed
    pas_vars['nap_consumed_fdp6zt'] = f_vars2np(lp_vars, 'v_nap_consumed', fdp6zt, keys_z, z_pos=-2)

    ##poc consumed
    pas_vars['poc_consumed_fp6lz'] = f_vars2np(lp_vars, 'v_poc', fp6lz, keys_z, z_pos=-1)

    return pas_vars


def f_stock_cash_summary(lp_vars, r_vals):
    '''
    Returns:

        #. expense and revenue items

    '''
    ##get reshaped variable
    stock_vars = f_stock_reshape(lp_vars, r_vals)

    ##keys
    keys_c = r_vals['fin']['keys_c']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_k = r_vals['pas']['keys_k']

    ##numbers
    sire_numbers_zg0 = stock_vars['sire_numbers_zg0']
    dams_numbers_k2tvanwziy1g1 = stock_vars['dams_numbers_k2tvanwziy1g1']
    prog_numbers_k3k5twzia0xg2 = stock_vars['prog_numbers_k3k5twzia0xg2']
    offs_numbers_k3k5tvnwziaxyg3 = stock_vars['offs_numbers_k3k5tvnwziaxyg3']

    ##husb cost
    sire_cost_czg0 = r_vals['stock']['sire_cost_czg0'] * sire_numbers_zg0
    dams_cost_k2ctva1nwziyg1 = r_vals['stock']['dams_cost_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    offs_cost_k3k5ctvnwziaxyg3 = r_vals['stock']['offs_cost_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ##purchase cost
    sire_purchcost_czg0 = r_vals['stock']['purchcost_sire_cg0'] * sire_numbers_zg0

    ##sale income
    salevalue_czg0 = r_vals['stock']['salevalue_czg0'] * sire_numbers_zg0
    salevalue_k2ctva1nwziyg1 = r_vals['stock']['salevalue_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    salevalue_k3k5ctwzia0xg2 = r_vals['stock']['salevalue_k3k5ctwzia0xg2'] * prog_numbers_k3k5twzia0xg2[:, :, na,
                                                                                     ...]
    salevalue_k3k5ctvnwziaxyg3 = r_vals['stock']['salevalue_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ##wool income
    woolvalue_czg0 = r_vals['stock']['woolvalue_czg0'] * sire_numbers_zg0
    woolvalue_k2ctva1nwziyg1 = r_vals['stock']['woolvalue_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    woolvalue_k3k5ctvnwziaxyg3 = r_vals['stock']['woolvalue_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ###sum axis to return total income in each cash period
    siresale_cz = fun.f_reduce_skipfew(np.sum, salevalue_czg0, preserveAxis=(0,1))  # sum all axis except c
    damssale_cz = fun.f_reduce_skipfew(np.sum, salevalue_k2ctva1nwziyg1, preserveAxis=(1,7))  # sum all axis except c
    progsale_cz = fun.f_reduce_skipfew(np.sum, salevalue_k3k5ctwzia0xg2, preserveAxis=(2,5))  # sum all axis except c
    offssale_cz = fun.f_reduce_skipfew(np.sum, salevalue_k3k5ctvnwziaxyg3, preserveAxis=(2,7))  # sum all axis except c
    sirewool_cz = fun.f_reduce_skipfew(np.sum, woolvalue_czg0, preserveAxis=(0,1))  # sum all axis except c
    damswool_cz = fun.f_reduce_skipfew(np.sum, woolvalue_k2ctva1nwziyg1, preserveAxis=(1,7))  # sum all axis except c
    offswool_cz = fun.f_reduce_skipfew(np.sum, woolvalue_k3k5ctvnwziaxyg3, preserveAxis=(2,7))  # sum all axis except c
    stocksale_cz = siresale_cz + damssale_cz + progsale_cz + offssale_cz
    wool_cz = sirewool_cz + damswool_cz + offswool_cz

    sirecost_cz = fun.f_reduce_skipfew(np.sum, sire_cost_czg0, preserveAxis=(0,1))  # sum all axis except c
    damscost_cz = fun.f_reduce_skipfew(np.sum, dams_cost_k2ctva1nwziyg1, preserveAxis=(1,7))  # sum all axis except c
    offscost_cz = fun.f_reduce_skipfew(np.sum, offs_cost_k3k5ctvnwziaxyg3, preserveAxis=(2,7))  # sum all axis except c

    sire_purchcost_cz = fun.f_reduce_skipfew(np.sum, sire_purchcost_czg0, preserveAxis=(0,1))  # sum all axis except c

    ##expenses sup feeding
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    sup_grain_cost_cz = grain_summary['sup_exp_c_z'].reindex(keys_c) #get index into correct order
    grain_fed_kp6_z = f_grain_sup_summary(lp_vars, r_vals, option=2)
    supp_feedstorage_cost_ckp6_z = r_vals['sup']['total_sup_cost_ckp6_z']
    grain_fed_ckp6_z = grain_fed_kp6_z.unstack().reindex(supp_feedstorage_cost_ckp6_z.unstack().index,axis=0,level=1).stack()
    supp_feedstorage_cost_ckp6_z = supp_feedstorage_cost_ckp6_z.mul(grain_fed_ckp6_z)
    supp_feedstorage_cost_c_z = supp_feedstorage_cost_ckp6_z.sum(level=(0))
    supp_feedstorage_cost_cz = supp_feedstorage_cost_c_z.reindex(keys_c).values #to get c axis in correct order (because is was sorted alphabetically)

    ##infrastructure
    fixed_infra_cost_c = np.sum(r_vals['stock']['rm_stockinfra_fix_h1c'], axis=0)
    var_infra_cost_cz = np.sum(r_vals['stock']['rm_stockinfra_var_h1c'][...,na] * stock_vars['infrastructure_h1z'][:,na,:], axis=0)
    total_infra_cost_cz = fixed_infra_cost_c[:,na] + var_infra_cost_cz

    ##total costs
    stockcost_cz = (sirecost_cz + damscost_cz + offscost_cz + sup_grain_cost_cz.values + total_infra_cost_cz
                    + supp_feedstorage_cost_cz + sire_purchcost_cz)

    return stocksale_cz, wool_cz, stockcost_cz


def f_labour_summary(lp_vars, r_vals, option=0):
    '''
    :param option:

        #. return total labour cost
        #. return amount for each enterprise

    '''

    ##shapes
    keys_c = r_vals['fin']['keys_c']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_z = r_vals['stock']['keys_z']
    len_c = len(keys_c)
    len_p5 = len(keys_p5)
    len_z = len(keys_z)

    cas_shape = len_p5, len_z

    ##total labour cost
    if option == 0:
        ###casual
        quantity_casual_p5z = f_vars2np(lp_vars, 'v_quantity_casual', cas_shape,keys_z, z_pos=-1)
        casual_cost_p5zc = r_vals['lab']['casual_cost_p5zc']
        cas_cost_zc = np.sum(casual_cost_p5zc * quantity_casual_p5z[...,na], axis=0)
        ###perm
        quantity_perm_z = f_vars2np(lp_vars, 'v_quantity_perm', len_z, keys_z, z_pos=-1)
        perm_cost_c = r_vals['lab']['perm_cost_c']
        perm_cost_zc = perm_cost_c * quantity_perm_z[...,na]
        ###manager
        quantity_manager_z = f_vars2np(lp_vars, 'v_quantity_manager', len_z, keys_z, z_pos=-1)
        manager_cost_c = r_vals['lab']['manager_cost_c']
        manager_cost_zc = manager_cost_c * quantity_manager_z[...,na]
        ###total
        total_lab_cost_zc = cas_cost_zc + perm_cost_zc + manager_cost_zc
        return total_lab_cost_zc
    ##labour breakdown for each worker level (table: labour period by worker level)
    if option == 1:
        ###sheep
        manager_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_manager']).unstack()
        prem_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_permanent']).unstack()
        casual_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_casual']).unstack()
        sheep_labour = pd.concat([manager_sheep_p5w, prem_sheep_p5w, casual_sheep_p5w], axis=1).sum(axis=1, level=0)
        ###crop
        manager_crop_p5w = pd.Series(lp_vars['v_crop_labour_manager']).unstack()
        prem_crop_p5w = pd.Series(lp_vars['v_crop_labour_permanent']).unstack()
        casual_crop_p5w = pd.Series(lp_vars['v_crop_labour_casual']).unstack()
        crop_labour = pd.concat([manager_crop_p5w, prem_crop_p5w, casual_crop_p5w], axis=1).sum(axis=1, level=0)
        ###fixed
        manager_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_manager']).unstack()
        prem_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_permanent']).unstack()
        casual_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_casual']).unstack()
        fixed_labour = pd.concat([manager_fixed_p5w, prem_fixed_p5w, casual_fixed_p5w], axis=1).sum(axis=1, level=0)
        return sheep_labour, crop_labour, fixed_labour


def f_dep_summary(lp_vars, r_vals):
    ##depreciation total
    keys_z = r_vals['stock']['keys_z']
    len_z = len(keys_z)
    dep_z = f_vars2np(lp_vars, 'v_dep', len_z, keys_z, z_pos=-1)
    return dep_z

def f_minroe_summary(lp_vars, r_vals):
    ##min return on expense cost
    keys_z = r_vals['stock']['keys_z']
    minroe_z = f_vars2df(lp_vars, 'v_minroe', keys_z).droplevel(1) #drop level 1 because no sets therefore nan
    return minroe_z

def f_asset_value_summary(lp_vars, r_vals):
    ##asset opportunity cost
    keys_z = r_vals['stock']['keys_z']
    asset_value_z = f_vars2df(lp_vars, 'v_asset', keys_z).droplevel(1) #drop level 1 because no sets therefore nan
    return asset_value_z

def f_overhead_summary(r_vals):
    ##overheads/fixed expenses
    exp_fix_c = r_vals['fin']['overheads']
    return exp_fix_c

#todo this should probably report z as an index rather than summing it.
def f_dse(lp_vars, r_vals, method, per_ha, summary=False):
    '''
    DSE calculation.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param method: int

            0. dse by normal weight
            1. dse by mei

    :param per_ha: Bool
        if true it returns DSE/ha else it returns total dse
    :param summary: Bool
        if true it returns the total DSE/ha in fp0
    :return DSE per pasture hectare for each sheep group.
    '''
    ##keys for table that is reported
    keys_p6 = r_vals['stock']['keys_p6']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']

    ##user can change this if they want to report different axis. Keys must be a list and axis must be tuple. Check names below to get the axis positions.
    sire_preserve_ax = 0
    sire_key = [keys_p6]
    dams_preserve_ax = (1, 3)
    dams_key = [keys_p6, keys_v1]
    offs_preserve_ax = (2, 4)
    offs_key = [keys_p6, keys_v3]

    if summary:
        sire_preserve_ax = 0
        sire_key = [keys_p6]
        dams_preserve_ax = 1
        dams_key = [keys_p6]
        offs_preserve_ax = 2
        offs_key = [keys_p6]

    stock_vars = f_stock_reshape(lp_vars, r_vals)

    if method == 0:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_zg0'] * r_vals['stock']['dsenw_p6zg0'], preserveAxis=sire_preserve_ax)  # sum all axis except preserveAxis
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_k2tvanwziy1g1'][:, na, ...]
                                        * r_vals['stock']['dsenw_k2p6tva1nwziyg1'], preserveAxis=dams_preserve_ax)  # sum all axis except preserveAxis
        ##offs
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_k3k5tvnwziaxyg3'][:, :, na, ...] * r_vals['stock'][
            'dsenw_k3k5p6tvnwziaxyg3'], preserveAxis=offs_preserve_ax)  # sum all axis except preserveAxis
    else:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_zg0'] * r_vals['stock']['dsemj_p6zg0'], preserveAxis=sire_preserve_ax)  # sum all axis except preserveAxis
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_k2tvanwziy1g1'][:, na, ...] * r_vals['stock'][
            'dsemj_k2p6tva1nwziyg1'], preserveAxis=dams_preserve_ax)  # sum all axis except preserveAxis
        ##offs
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_k3k5tvnwziaxyg3'][:, :, na, ...] * r_vals['stock'][
            'dsemj_k3k5p6tvnwziaxyg3'], preserveAxis=offs_preserve_ax)  # sum all axis except preserveAxis

    ##dse per ha if user opts for this level of detail
    if per_ha:
        pasture_area = f_area_summary(lp_vars, r_vals, option=2)
        dse_sire = fun.f_divide(dse_sire, pasture_area)
        dse_dams = fun.f_divide(dse_dams, pasture_area)
        dse_offs = fun.f_divide(dse_offs, pasture_area)

    ##turn to table - rows and cols need to be a list of lists/arrays
    dse_sire = fun.f_produce_df(dse_sire.ravel(), rows=sire_key, columns=[['Sire DSE']])
    dse_dams = fun.f_produce_df(dse_dams.ravel(), rows=dams_key, columns=[['Dams DSE']])
    dse_offs = fun.f_produce_df(dse_offs.ravel(), rows=offs_key, columns=[['Offs DSE']])

    if summary:
        return (dse_sire.iloc[0, 0] + dse_dams.iloc[0, 0] + dse_offs.iloc[0, 0]).round(2)  #sum SR for all sheep groups in FP0 (to return winter sr)
    else:
        return dse_sire, dse_dams, dse_offs


def f_profitloss_table(lp_vars, r_vals):
    '''
    Returns profit and loss statement for selected trials. Multiple trials result in a stacked pnl table.

    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :return: dataframe

    '''
    ##read stuff from other functions that is used in rev and cost section
    exp_fert_k_cz, exp_chem_k_cz, misc_exp_k_cz, rev_grain_k_cz = f_crop_summary(lp_vars, r_vals, option=0)
    exp_mach_c_zk, mach_insurance_c = f_mach_summary(lp_vars, r_vals)
    stocksale_cz, wool_cz, stockcost_cz = f_stock_cash_summary(lp_vars, r_vals)
    ##other info required below
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    keys_c = r_vals['fin']['keys_c']

    ##create p/l dataframe
    idx = pd.IndexSlice
    keys_z = r_vals['stock']['keys_z']
    subtype_rev = ['grain', 'sheep sales', 'wool', 'Total Revenue']
    subtype_exp = ['crop', 'pasture', 'stock', 'machinery', 'labour', 'fixed', 'Total expenses']
    subtype_tot = ['asset_value', 'depreciation', 'minRoe', 'EBITD', 'Interest', 'obj']
    pnl_rev_index = pd.MultiIndex.from_product([keys_z, ['Revenue'], subtype_rev], names=['Season', 'Type', 'Subtype'])
    pnl_exp_index = pd.MultiIndex.from_product([keys_z, ['Expense'], subtype_exp], names=['Season', 'Type', 'Subtype'])
    pnl_tot_index = pd.MultiIndex.from_product([keys_z, ['Total'], subtype_tot], names=['Season', 'Type', 'Subtype'])
    pnl_index = pnl_rev_index.append(pnl_exp_index).append(pnl_tot_index)
    pnl = pd.DataFrame(index=pnl_index, columns=keys_c)  # need to initialise df with multiindex so rows can be added

    ##income
    rev_grain_zc = rev_grain_k_cz.sum(axis=0).unstack(0)  # sum landuse axis
    ###add to p/l table each as a new row
    pnl.loc[idx[:,'Revenue','grain'],:] = rev_grain_zc.reindex(keys_c, axis=1).values #can't just assign values because c axis has been sorted alphabetically so need to put back in correct order for cashflow
    pnl.loc[idx[:, 'Revenue', 'sheep sales'], :] = stocksale_cz.T
    pnl.loc[idx[:, 'Revenue', 'wool'], :] = wool_cz.T
    pnl.loc[idx[:, 'Revenue', 'Total Revenue'], :] = pnl.loc[pnl.index.get_level_values(1) == 'Revenue'].sum(axis=0,level=0).values

    ##expenses
    ####machinery
    mach_c_z = exp_mach_c_zk.sum(axis=1,level=0)  # sum landuse
    mach_c_z = mach_c_z.add(mach_insurance_c, axis=0)
    ####crop & pasture
    pasfert_c_z = exp_fert_k_cz[exp_fert_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    cropfert_c_z = exp_fert_k_cz[~exp_fert_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    paschem_c_z = exp_chem_k_cz[exp_chem_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    cropchem_c_z = exp_chem_k_cz[~exp_chem_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    pasmisc_c_z = misc_exp_k_cz[misc_exp_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    cropmisc_c_z = misc_exp_k_cz[~misc_exp_k_cz.index.isin(all_pas)].sum(axis=0).unstack()
    pas_c_z = pd.concat([pasfert_c_z, paschem_c_z, pasmisc_c_z], axis=0).sum(axis=0, level=0)
    crop_c_z = pd.concat([cropfert_c_z, cropchem_c_z, cropmisc_c_z], axis=0).sum(axis=0, level=0)
    ####labour
    labour_zc = f_labour_summary(lp_vars, r_vals, option=0)
    ####fixed overhead expenses
    exp_fix_c = f_overhead_summary(r_vals)
    exp_fix_cz = pd.concat([exp_fix_c] * len(keys_z),axis=1).values
    ###add to p/l table each as a new row
    pnl.loc[idx[:, 'Expense', 'crop'], :] = crop_c_z.T.reindex(keys_c, axis=1).values
    pnl.loc[idx[:, 'Expense', 'pasture'], :] = pas_c_z.T.reindex(keys_c, axis=1).values
    pnl.loc[idx[:, 'Expense', 'stock'], :] = stockcost_cz.T
    pnl.loc[idx[:, 'Expense', 'machinery'], :] = mach_c_z.T.reindex(keys_c, axis=1).values
    pnl.loc[idx[:, 'Expense', 'labour'], :] = labour_zc
    pnl.loc[idx[:, 'Expense', 'fixed'], :] = exp_fix_cz.T
    pnl.loc[idx[:, 'Expense', 'Total expenses'], :] = pnl.loc[pnl.index.get_level_values(1) == 'Expense'].sum(axis=0,level=0).values

    ##EBIT
    ebitd = (pnl.loc[idx[:, 'Revenue', 'Total Revenue']] - pnl.loc[idx[:, 'Expense', 'Total expenses']]).values
    pnl.loc[idx[:, 'Total', 'EBITD'], :] = ebitd

    ##interest - note this is debit (currently debit and credit are the same if this changes the calc below will need to be modified)
    interest = r_vals['fin']['interest_rate']
    mo_interest = np.zeros(ebitd.shape)
    for i in range(ebitd.shape[-1] - 1): #-1 because last period gets no interest.
        cum_cash = np.sum(ebitd[:,0:i+1])
        cum_interest = np.sum(mo_interest[:,0:i+1])
        mo_interest[:,i] = (interest-1) * (cum_cash + cum_interest)
    pnl.loc[idx[:, 'Total', 'Interest'], :] = mo_interest

    ##add a column which is total of all cashflow period
    pnl['Full year'] = pnl.sum(axis=1)

    ##intrest, depreciation asset opp and minroe
    ###depreciation
    dep_z = f_dep_summary(lp_vars, r_vals)
    ###minroe
    minroe_z = f_minroe_summary(lp_vars,r_vals)
    ###asset opportunity cost
    asset_value_z = f_asset_value_summary(lp_vars,r_vals)

    ##add the assets & minroe & depreciation
    pnl.loc[idx[:, 'Total', 'depreciation'],'Full year'] = dep_z
    pnl.loc[idx[:, 'Total', 'asset_value'],'Full year'] = asset_value_z.values
    pnl.loc[idx[:, 'Total', 'minRoe'],'Full year'] = minroe_z.values

    ##add the objective
    pnl.loc[idx[:, 'Total', 'obj'],'Full year'] = f_profit(lp_vars, r_vals, option=2).values

    ##round numbers in df
    pnl = pnl.astype(float).round(1)  # have to go to float so rounding works

    ##sort the season level of index
    pnl = pnl.sort_index(axis=0, level=0)

    return pnl


def f_profit(lp_vars, r_vals, option=0):
    '''returns profit
    0- rev - (exp + minroe + asset_opp +dep). This is the model obj.
    1- rev - (exp + dep)
    2- same as 0 but reported for each season
    3- same as 1 but reported for each season
    '''
    keys_z = r_vals['stock']['keys_z']
    prob_z =r_vals['stock']['prob_z']
    obj_profit_z = f_vars2df(lp_vars, 'scenario_profit', keys_z).droplevel(1) #drop level 1 because no sets therefore nan
    minroe_z = f_minroe_summary(lp_vars, r_vals)
    asset_value_z = f_asset_value_summary(lp_vars, r_vals)
    if option == 0:
        return lp_vars['profit']
    elif option==1:
        minroe = sum(minroe_z * prob_z)
        asset_value = sum(asset_value_z * prob_z)
        return lp_vars['profit'] + minroe + asset_value
    elif option == 2:
        return obj_profit_z
    elif option==3:
        return obj_profit_z + minroe_z + asset_value_z


def f_stock_pasture_summary(lp_vars, r_vals, build_df=True, keys=None, type=None, index=[], cols=[], arith=0,
                            prod=1, na_prod=[], weights=None, na_weights=[], axis_slice={},
                            na_denweights=[], den_weights=1, na_prod_weights=[], prod_weights=1):
    '''

    ..Note::

        #. prod and weights must be broadcastable.
        #. Specify axes the broadcasted/expanded version.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param build_df: bool: return df
    :key type: str: either 'stock' or 'pas' to indicate calc type
    :key key: str: dict key for the axis keys
    :key index (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :key arith (optional, default = 0): int: arithmetic operation used.
                option 0: return production param averaged across all axis that are not reported.
                option 1: return weighted average of production param (using denominator weight return production per day the animal is on hand)
                option 2: weighted total production summed across all axis that are not reported.
                option 3: weighted total production for each  (axis not reported are disregarded)
                option 4: return weighted average of production param using prod>0 as the weights
                option 5: return the maximum value across all axis that are not reported.
    :key prod (optional, default = 1): str/int/float: if it is a string then it is used as a key for stock_vars, if it is an number that number is used as the prod value
    :key na_prod (optional, default = []): list: position to add new axis
    :key weights (optional, default = None): str: weights to be used in arith (typically a lp variable eg numbers). Only required when arith>0
    :key na_weights (optional, default = []): list: position to add new axis
    :key den_weights (optional, default = 1): str: key to variable used to weight the denominator in the weighted average (required p6 reporting)
    :key na_denweights (optional, default = []): list: position to add new axis
    :key prod_weights (optional, default = 1): str: keys to r_vals referencing array used to weight production.
    :key na_prod_weights (optional, default = []): list: position to add new axis
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: summary of a numpy array in a pandas table.
    '''
    keys_key = keys

    ##read from stock reshape function
    if type == 'stock':
        vars = f_stock_reshape(lp_vars, r_vals)
        r_vals = r_vals['stock']
        ###keys that will become the index and cols for table
        keys = r_vals[keys_key]
    else:
        vars = f_pasture_reshape(lp_vars, r_vals)
        r_vals = r_vals['pas']
        ###keys that will become the index and cols for table
        keys = vars[keys_key]

    ##if no weights then make None
    try:
        weights = vars[weights]
    except KeyError:
        weights = None

    ##initilise prod array from either r_vals or default value (this means you can preform arith with any number - mainly used for pasture when there is no production param)
    if isinstance(prod, str):
        prod = r_vals[prod]
    else:
        prod = np.array([prod])

    ##initilise prod_weight array from either r_vals or default value
    if isinstance(prod_weights, str):
        prod_weights = r_vals[prod_weights]
    else:
        prod_weights = np.array([prod_weights])

    ##den weight - used in weighted average calc (default is 1)
    if isinstance(den_weights, str):
        den_weights = r_vals[den_weights]

    ##other manipulation
    prod, weights, den_weights, prod_weights = f_add_axis(prod, na_prod, prod_weights, na_prod_weights, weights, na_weights, den_weights, na_denweights)
    prod, prod_weights, weights, den_weights, keys = f_slice(prod, prod_weights, weights, den_weights, keys, arith, axis_slice)
    ##preform arith. if an axis is not reported it is included in the arith and the axis disappears
    report_idx = index + cols
    arith_axis = list(set(range(len(prod.shape))) - set(report_idx))
    prod = f_arith(prod, prod_weights, weights, den_weights, arith, arith_axis)
    ##check for errors
    f_numpy2df_error(prod, weights, arith_axis, index, cols)
    if build_df:
        prod = f_numpy2df(prod, keys, index, cols)
        return prod
    else:
        return prod, keys


def f_lambing_status(lp_vars, r_vals, option=0, keys=None, index=[], cols=[], axis_slice={}):
    '''
    Depending on the option selected this function can calc:
        Lamb survival (per ewe at start of dvp when lambing occurs - eg mort is included)
        Weaning %  (per dam at the start of the dvp when mating occurs - eg mort is included)
        Scanning %
        Proportion of dry ewes

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :key option (optional, default = 0): int:
            option 0: survival %
            option 1: wean %
            option 2: scan %
            option 3: Proportion of dry ewes
    :key index (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :key arith_axis (optional, default = []): list: axis to preform arithmetic operation along.
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: pandas df
    '''



    ##params for specific options
    type = 'stock'
    if option == 0:
        prod = 'nyatf_birth_k2tva1e1b1nw8ziyg1'
        prod2 = 'nfoet_birth_k2tva1e1b1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [4,5]
        keys = 'dams_keys_k2tvaeb9nwziy1g1'

    elif option == 1:
        prod = 'nyatf_wean_k2tva1nw8ziyg1'
        prod2 = 'n_mated_k2tva1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = []
        keys = 'dams_keys_k2tvanwziy1g1'

    elif option == 2:
        prod = 'nfoet_scan_k2tva1nw8ziyg1'
        prod2 = 'n_mated_k2tva1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = []
        keys = 'dams_keys_k2tvanwziy1g1'

    elif option == 3:
        prod = 'n_drys_k2tva1nw8ziyg1'
        prod2 = 'n_mated_k2tva1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = []
        keys = 'dams_keys_k2tvanwziy1g1'

    ##params for all options
    arith = 2

    ##colate the lp and report vals using f_stock_pasture_summary
    numerator, keys_sliced = f_stock_pasture_summary(lp_vars, r_vals, build_df=False, type=type, prod=prod, weights=weights,
                           na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    denominator, keys_sliced = f_stock_pasture_summary(lp_vars, r_vals, build_df=False, type=type, prod=prod2, weights=weights,
                           na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)

    ##calcs for survival
    if option == 0:
        prog_alive_k2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(numerator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7) #b9 axis is shorten b axis: [0,1,2,3]
        prog_born_k2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(denominator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7)
        percentage = fun.f_divide(prog_alive_k2tvpa1e1b1nw8ziyg1, prog_born_k2tvpa1e1b1nw8ziyg1)

    ##calc for wean % or scan %
    else:
        percentage= fun.f_divide(numerator, denominator)

    ##make table
    percentage = f_numpy2df(percentage, keys_sliced, index, cols)
    return percentage



############################
# functions for numpy arrays#
############################

def f_numpy2df_error(prod, weights, arith_axis, index, cols):
    ##error handle 1: can't preform arithmetic along an axis and also report that axis and the index or col
    arith_occur = len(arith_axis) >= 1
    arith_error = any(item in index for item in arith_axis) or any(item in cols for item in arith_axis)
    if arith_occur and arith_error:  # if arith is happening and there is an error in selected axis
        raise exc.ArithError('''Arith error: can't preform operation along an axis that is going to be reported as the index or col''')

    ##error handle 2: once arith has been completed all axis that are not singleton must be used in either the index or cols
    if arith_occur:
        nonzero_idx = arith_axis + index + cols  # join lists
    else:
        nonzero_idx = index + cols  # join lists
    error = [prod.shape.index(size) not in nonzero_idx for size in prod.shape if size > 1]
    if any(error):
        raise exc.AxisError('''Axis error: active axes exist that are not used in arith or being reported as index or columns''')

    ##error 3: preforming arith with no weights
    if arith_occur and weights is None:
        raise exc.ArithError('''Arith error: weights are not included''')
    return


def f_add_axis(prod, na_prod, prod_weights, na_prod_weights, weights, na_weights, den_weights, na_denweights):
    '''
    Adds new axis if required.

    :param weights: array
    :param na_weights: list: position to add new axis
    :param prod: array
    :param na_prod: list: position to add new axis
    :return: expanded array
    '''
    weights = np.expand_dims(weights, na_weights)
    den_weights = np.expand_dims(den_weights, na_denweights)
    prod = np.expand_dims(prod, na_prod)
    prod_weights = np.expand_dims(prod_weights, na_prod_weights)
    return prod, weights, den_weights, prod_weights


def f_slice(prod, prod_weights, weights, den_weights, keys, arith, axis_slice):
    '''
    Slices the prod, weights and key arrays

    :param prod: array: production param
    :param prod_weights: array: production param weights
    :param weights: array: weights (typically the variable associated with the prod param)
    :param keys: list: keys for axes
    :param axis_slice: dict: containing list of with slice params (start, stop, step)
    :return: prod array
    '''
    ##slice axis - slice the keys and the array - if user hasn't specified slice the whole axis will be included
    sl = [slice(None)] * prod.ndim
    for axis, slc in axis_slice.items():
        start = slc[0]
        stop = slc[1]
        step = slc[2]
        sl[axis] = slice(start, stop, step)
        keys[axis] = keys[axis][start:stop:step]
    ###apply slice to np array
    prod, prod_weights, weights, den_weights = np.broadcast_arrays(prod, prod_weights, weights,
                                                     den_weights)  # if arith is being conducted these arrays need to be the same size so slicing can work
    prod = prod[tuple(sl)]
    prod_weights = prod_weights[tuple(sl)]
    weights = weights[tuple(sl)]
    den_weights = den_weights[tuple(sl)]
    return prod, prod_weights, weights, den_weights, keys


def f_arith(prod, prod_weights, weight, den_weights, arith, axis):
    '''
    option 0: return production param averaged across all axis that are not reported.
    option 1: return weighted average of production param (using denominator weight return production per day the animal is on hand)
    option 2: weighted total production summed across all axis that are not reported.
    option 3: weighted total production for each  (axis not reported are disregarded)
    option 4: return weighted average of production param using prod>0 as the weights
    option 5: return the maximum value across all axis that are not reported.

    :param prod: array: production param
    :param prod_weight: array: weights the production param
    :param weight: array: weights (typically the variable associated with the prod param)
    :param den_weight: array: weights the denominator in the weighted average calculation
    :param arith: int: arith option
    :param axis: list: axes to preform arith along
    :return: array
    '''
    ##adjust prod by prod_weights
    prod = prod * prod_weights
    ##calc if keep dims
    keepdims = len(axis) != len(prod.shape)
    ##option 0
    if arith == 0:
        prod = np.mean(prod, tuple(axis), keepdims=keepdims)
    ##option 1
    if arith == 1:
        prod = fun.f_weighted_average(prod, weight, tuple(axis), keepdims=keepdims, den_weights=den_weights)
    ##option 2
    if arith == 2:
        prod = np.sum(prod * weight, tuple(axis), keepdims=keepdims)
    ##option 3
    if arith == 3:
        prod = prod * weight
    ##option 4
    if arith == 4:
        prod = fun.f_divide(np.sum(prod * (prod>0), tuple(axis), keepdims=keepdims), np.sum(prod>0, tuple(axis), keepdims=keepdims))
    ##option 5
    if arith == 5:
        prod = np.max(prod, tuple(axis), keepdims=keepdims)

    return prod


def f_numpy2df(prod, keys, index, cols):
    if prod.size <= 1 and prod.ndim <= 1:
        return pd.DataFrame([prod])  # don't need to reshape etc if everything is summed and prod is just one number
    ##move x axis to front
    dest = list(range(len(index)))
    prod = np.moveaxis(prod, index, dest)

    ##move y axis to front behind x axis (note if an axis is not an index or col then it should be singleton)
    np_cols_y = np.array(cols)
    np_cols_xy = np_cols_y[na]
    np_index_xy = np.array(index)[:, na]
    cols_adj = np.sum(np_index_xy > np_cols_xy, axis=0)
    np_cols_y = np_cols_y + cols_adj
    dest = list(range(len(index), len(index) + len(cols)))
    prod = np.moveaxis(prod, np_cols_y, dest)

    ##select keys
    x_keys = []
    x_len = 1
    for axis in index:
        x_len *= len(keys[axis])
        x_keys.append(keys[axis])
    y_keys = []
    y_len = 1
    for axis in cols:
        y_len *= len(keys[axis])
        y_keys.append(keys[axis])

    ##reshape
    prod = prod.reshape(x_len, y_len)

    ##make df
    prod = fun.f_produce_df(prod, x_keys, y_keys)

    return prod
