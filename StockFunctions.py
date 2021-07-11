"""
author: Young


Summary
--------
The stock module generates production parameters for livestock under a range of different conditions. These are
represented as different decision variables which allows the optimisation of a wide range of management decisions.

Data is generated for the following components:

    * Animal live weight and sale values.
    * Fleece production data, both quantity and quality (including the impact of ewe live weight profile during
      pregnancy on the lifetime performance of the progeny).
    * Dam reproduction levels.
    * Mortality rates of dams, progeny and dry animals.
    * Energy requirement profile and the nutritive value to achieve the target whole body energy profiles.
    * Foetal growth rates and birth weights for progeny.
    * Milk production and progeny weaning weights.
    * Husbandry cost and labour requirements.

.. note:: The livestock component of the sub-model currently only represents sheep, however, the option of
          representing cattle exists with little code alteration, simply addition of management inputs for the cattle.

Structure
---------
The livestock data generating sub-model is a simulation model that predicts feed intake, energy requirements,
production, emissions, husbandry and labour requirements during the lifetime of different classes of stock offered
different amounts of feed. The above values are calculated for each ‘class’ of stock represented. The feed supply
offered to the animals is not based on simulating a growing pasture, rather the FOO, digestibility and supplement
offered are inputs to the data generator. The outcome is the parameters required to define the production
possibilities that are included in the pyomo matrix of the AFO model.

The data generator model simulates the sires, dams and offspring from weaning to their latest possible sale age
and simulates the young at foot from birth to weaning. The initial animal for the sires, dams and offspring are
based on input values for liveweight, cfw and fibre diameter.

The prediction equations included in the data generator can be selected from any of a number of equation sources:

    #. GrazPlan equations as documented in Freer et al. 2012, which are an improved version of the
       Australian Feed Standards (SCA 1990).
    #. Research trials carried out by Murdoch University, DPIRD & DPI Victoria that have quantified
       the impact of changing nutrition on production. This research began with the Lifetime Wool Trial (refs here) but has continued with a suite of other projects (refs here)
    #. A selection of other sources that have developed equations to predict animal performance. Including:

        a. Blaxter & Clapperton for enteric methane emissions
        b. NGGI for emissions including methane and nitrous oxide
        c. Hutton Oddy’s group for alternative equations for heat production associated with maintenance and liveweight gain.

The relationships are calculated for a range of age groups for each of the following groups:

    #. Sires - entire males used for breeding purposes.
    #. Dams - females of reproductive age, mated or unmated.
    #. Young at foot - male and female progeny prior to weaning.
    #. Offspring - female or male progeny of the dams from post weaning until sale.

For each animal group there are a range of classes represented and for each class the range of optimised
management decisions are included. The classes and the management decisions are represented as axes (or dimensions)
of a multi-dimensional numpy array. Note: numpy, like python, works on a base 0 system so the numbering of the
slices of each axis begins at 0. The array axes are:

The management optimisation structure
-------------------------------------
To disaggregate the matrix the livestock management options are grouped into decision variable periods (DVP). At a
DVP junction the animals are distributed from the decision variables of the previous period to the decision
variables of the next generator period. This is necessary when the sheep change clusters, for example at scanning
the pregnancy status of the dam is identified and if it is a separate activity it can then be managed differently
to a dry dam, whereas prior to scanning they are clustered and must be managed the same. The timing of the DVPs
is based on periods when the management groups change, this includes pre-joining, scanning and birth.
Animal sale options can be defined for each DVP. The selling opportunity is represented by the t axis and
(in version 1) there is opportunity for a maximum of 2 sale time per DVP for dams and offspring. Although
these selling opportunities can be masked from the matrix if sales in a particular DVP is not desired.

For the dams the annual cycle starts with prejoining and the other DVPs are based on the reproduction calendar:
early pregnancy (period_between_prejoinscan), late pregnancy (period_between_scanbirth), lactation and post weaning
recovery (period_between_birthprejoin). The Dams sale time is just after the main shearing, and additionally dry
ewes can be sold after scanning.

For the offspring (dry stock) the annual cycle starts with shearing. The offspring only require one DVP
for the year because the animals do not change class and require distributing during the year as do the dams.
The 2 sale options for Offspring can be input as a combination of either a target date or a target liveweight
and the animals can be shorn at sale (retained animals are shorn at the main shearing).

The nutrition optimisation structure
------------------------------------
Representing different nutrition levels is useful in analyses where it is expected that the technique or
technology being tested will alter the optimal feed supply to the animals. This may be because the technology
alters the feed supply or the production efficiency of the animals, either of which may alter optimal feed allocation.

Variable liveweight patterns also allows realistic evaluation of nutrition strategies that don’t include the
implicit assumption that managers can ration the intake of their livestock. The model can be constrained to
represent animals grazing to their voluntary feed intake (by making the volume constraint an “E” constraint)
and feasible options exist that include increasing animal intake.

The range of nutrition levels are represented by profiles that are continuous for the entire year. At the end
of the nutrition cycle (year) the range of final liveweights are ‘condensed’ back to a range of starting weights
for the start of the next nutrition cycle.

A feed variation period (FVP) is a user defined period at the start of which the nutrition profiles can
diverge to different levels (eg. medium, high or low).The feed variation periods are defined to
represent time points when either:

    #. An alternative nutrition level may be optimal for different classes of stock.
    #. Quantifying the value of altering nutrition is desired for an analysis report.


"""
import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats
import math
import time


# from dateutil.relativedelta import relativedelta

import Functions as fun
import FeedsupplyFunctions as fsfun
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Sensitivity as sen

na=np.newaxis


def f1_sim_periods(start_year, periods_per_year, oldest_animal):
    '''
    Define the dates for the simulation periods.
    Starts on 1 Jan of the year with the earliest birthdate.

    Parameters:
    start_year = int: year to start simulation.
    periods_per_year = int: number of periods per year.
    oldest_animal = float: age of the oldest animal to be simulated (yrs).

    Returns:
    n_sim_periods
    array of period dates (1D periods)
    array of period end dates (1D periods) (date of the last day in the period)
    index of the periods (for pyomo)
    step - seconds in each period
    '''
    n_sim_periods = int(oldest_animal * periods_per_year)
    start_date = dt.date(year=start_year, month=1,day=1)
    step = pd.to_timedelta(365.25 / periods_per_year,'D')
    step = step.to_numpy().astype('timedelta64[s]')
    index_p = np.arange(n_sim_periods)
    date_start_p =  (np.datetime64(start_date) + (step * index_p)).astype('datetime64[D]') #astype day rounds the date to the nearest day
    date_end_p = (np.datetime64(start_date - dt.timedelta(days=1)) + (step * (index_p+1))).astype('datetime64[D]') #minus one day to get the last day in the period not the first day of the next period.
    return n_sim_periods, date_start_p, date_end_p, index_p, step


def f1_period_is_(period_is, date_array, date_start_p=0, date_array2 = 0, date_end_p=0):
    '''
    Parameters
    ----------
    period_is : string
        type of period is calc to return.
    date_start_p : datetime64[D]
        start date of each period (must have all axis).
    date_end_p : datetime64[D]
        end date of each period (must have all axis).
    date_array : datetime64[D]
        array of dates of interest eg mating dates.
    date_array2 : datetime64[D]
        array of end dates used to determine if period is between.

    Returns
    -------
    period_is: boolean array shaped like the date array with the addition of the p axis. This is true if a given date from date array is within the date of a given period and false if not.

    period_is_any: 1D boolean array shape of the period dates array. True if any of the dates in the date array fall into a given period.

    period_is_between: return true if a the period is between two dates (it is inclusive ie if an activity occurs during the period that period will be treated as between the two dates)
    '''
    if period_is == 'period_is':
        period_is=np.logical_and((date_array>=date_start_p) , (date_array<=date_end_p))
        return period_is
    if period_is == 'period_is_any':
        period_is=np.logical_and((date_array>=date_start_p) , (date_array<=date_end_p))
        period_is_any = np.any(period_is,axis=tuple(range(1,period_is.ndim)))
        return period_is_any
    if period_is == 'period_is_pre':
        period_is_pre=(date_array>date_end_p)
        return period_is_pre
    if period_is == 'period_is_between':
        period_is_between= np.logical_and((date_array<=date_end_p) , (date_array2>date_start_p))
        return period_is_between


###################################
#input and manipulation functions #
###################################

def f1_c2g(params_c2, y=0, var_pos=0, condition=None, axis=0, dtype=False):
    '''
    Parameters
    ----------
    params_c2 : array
        parameter array - input from excel.
    y : array
        sensitivity array for genetic merit.
    var_pos : int
        position of last axis when inserted into all axis.

    Returns
    -------
    param array for each genotype. Grouped by sheep group ie sire, offs, dams, yatf.

    '''

    ##these inputs are used for each param so they don't need to be passed into the function.
    a_c2_c0 = pinp.sheep['a_c2_c0']
    i_g3_inc = pinp.sheep['i_g3_inc']
    i_mul_g0_c0 = sinp.stock['i_mul_g0c0']
    i_mul_g1_c0 = sinp.stock['i_mul_g1c0']
    i_mul_g2_c0 = sinp.stock['i_mul_g2c0']
    i_mul_g3_c0 = sinp.stock['i_mul_g3c0']
    i_mask_g0g3 = sinp.stock['i_mask_g0g3']
    i_mask_g1g3 = sinp.stock['i_mask_g1g3']
    i_mask_g2g3 = sinp.stock['i_mask_g2g3']
    i_mask_g3g3 = sinp.stock['i_mask_g3g3']

    ##convert params from c2 to c0
    params_c2 = params_c2.astype(float) #this is so that blank cells are converted to nan not none type because none type can't be multiplied etc
    params_c0 = params_c2[...,a_c2_c0]
    ##add y axis
    ###if y is not numpy ie was read in as an int because it was a single cell, it needs to be converted
    if type(y) == int:
        y = np.asarray([y])
    ###y is a 2d array however currently it only has one slice so it is read in as a 1d array. so i need to add second array
    if y.ndim == 1 and params_c0.ndim != 1:
        y=y[...,na]
    ###apply y mask
    y=y[...,uinp.parameters['i_mask_y']]
    params_c0 = np.multiply(params_c0[...,na,:],  y[...,na]) #na here is to account for c0 axis
    ##get axis into correct position
    if var_pos != None or var_pos != 0:
        extra_axes = tuple(range((var_pos + 1), -2))
    else: extra_axes = ()
    allaxis_params_c0 = np.expand_dims(params_c0, axis = extra_axes)
    ##create mask g?c0
    mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
    mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
    mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
    mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
    ##create array with the proportion of each pure genotype required to have each actual genotype included in the analysis
    mul_sire_genotypes_g0c0 = i_mul_g0_c0[mask_sire_inc_g0]
    mul_dams_genotypes_g0c0 = i_mul_g1_c0[mask_dams_inc_g1]
    mul_yatf_genotypes_g0c0 = i_mul_g2_c0[mask_yatf_inc_g2]
    mul_offs_genotypes_g0c0 = i_mul_g3_c0[mask_offs_inc_g3]
    ##convert params from c0 to g. nansum required when the selected c0 info is not filled out ^may be an issue if params are missing and mixed breed sheep is selected because it won't catch the error
    param_sire=np.nansum(allaxis_params_c0[..., na, :] * mul_sire_genotypes_g0c0, axis = -1)
    param_dams=np.nansum(allaxis_params_c0[..., na, :] * mul_dams_genotypes_g0c0, axis = -1)
    param_yatf=np.nansum(allaxis_params_c0[..., na, :] * mul_yatf_genotypes_g0c0, axis = -1)
    param_offs=np.nansum(allaxis_params_c0[..., na, :] * mul_offs_genotypes_g0c0, axis = -1)
    ##apply mask if required
    if condition is not None: #see if condition exists
        if type(condition) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition= np.asarray([condition]) #convert to numpy if it is singular input
        ###apply mask
        param_sire = np.compress(condition, param_sire, axis)
        param_dams = np.compress(condition, param_dams, axis)
        param_yatf = np.compress(condition, param_yatf, axis)
        param_offs = np.compress(condition, param_offs, axis)
    ##assign dtype
    if dtype:
        param_sire = param_sire.astype(dtype)
        param_dams = param_dams.astype(dtype)
        param_yatf = param_yatf.astype(dtype)
        param_offs = param_offs.astype(dtype)

    return param_sire, param_dams, param_yatf, param_offs


def f1_g2g(array_g,group,left_pos=0,swap=False,right_pos=-1,left_pos2=0,right_pos2=-1, left_pos3=0, right_pos3=0,
          condition = None, axis = 0, condition2 = None, axis2 = 0, move=False, source=0, dest=1):
    '''
    Parameters
    ----------
    array_g : array
        parameter array - input from excel.
    group : TYPE
        DESCRIPTION.
    left_pos : int
        position of axis to the left of where the new axis will be added.
    swap : boolean, optional
        do you want to swap the first tow axis?. The default is False.
    right_pos : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    left_pos2 : int
        position of axis to the left of where the new axis will be added.
    right_pos2 : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    condition: boolean, optional
        mask used to slice given axis.
    axis: int, optional
        axis to apply mask to.

    *note: if adding two sets of new axis add from right to left (then the pos variables align)

    Returns
    -------
    Reshapes, swaps axis if required, expands and converts g array to the correct g slices.
    '''

    i_g3_inc = pinp.sheep['i_g3_inc']
    i_mask_g0g3 = sinp.stock['i_mask_g0g3']
    i_mask_g1g3 = sinp.stock['i_mask_g1g3']
    i_mask_g2g3 = sinp.stock['i_mask_g2g3']
    i_mask_g3g3 = sinp.stock['i_mask_g3g3']

    ##swap axis if necessary
    if swap:
        array_g = np.swapaxes(array_g, 0, 1)

    ##move axis if necessary
    if move:
        array_g = np.moveaxis(array_g, source=source, destination=dest)


    ##get axis into correct position 1
    if left_pos != None or left_pos != 0:
        extra_axes = tuple(range((left_pos + 1), right_pos))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)

    ##get axis into correct position 2 (some arrays need singleton axis added in multiple places ie separated by a used axis)
    if left_pos2 != None or left_pos2 != 0:
        extra_axes = tuple(range((left_pos2 + 1), right_pos2))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)

    ##get axis into correct position 3 (some arrays need singleton axis added in multiple places ie separated by a used axis)
    if left_pos3 != 0:
        extra_axes = tuple(range((left_pos3 + 1), right_pos3))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)

    ##select the required genotypes based on the offspring the user wants to model.
    if group == 'sire':
        ##create mask g?g
        mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_sire_inc_g0]
    elif group == 'dams':
        ##create mask g?g
        mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_dams_inc_g1]
    elif group == 'offs':
        ##create mask g?g
        mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_offs_inc_g3]
    elif group == 'yatf':
        ##create mask g?g
        mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_yatf_inc_g2]
    ##apply mask if required
    if condition is not None: #see if condition exists
        if type(condition) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition= np.asarray([condition]) #convert to numpy if it is singular input
            array = np.compress(condition, array, axis)
        else:
            array = np.compress(condition, array, axis)
    if condition2 is not None: #see if condition exists
        if type(condition2) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition2= np.asarray([condition2]) #convert to numpy if it is singular input
            array = np.compress(condition2, array, axis2)
        else:
            array = np.compress(condition2, array, axis2)
    return array


def f1_DSTw(scan_g, cycles=1):
    '''
    A numpy based calculation of the proportion of dry, single, twin & triplet bearing dams from a scanning percentage.
    Prediction is a polynomial formula y=intercept+ax+bx^2+cx^3+dx^4, where x is the scanning %

    Parameters
    ----------
    scan_g : np array - scanning percentage of genotypes if mated for the number of calibration cycles.
    cycles: int, optional - the number of cycles for which the prediction is required.
    Returns
    -------
    Proportion of dry, single, twins & triplets.

    '''
    calibration_cycles = 2  #The data used to calibrate the coefficients used are assumed to have been derived from mating for 2 cycles.

    ## predict the proportion of dry, single, twins & triplets if the dams were mated for the calibration period (2 cycles)
    scan_powers_s = uinp.sheep['i_scan_powers']  #scan powers are the exponential powers used in the polynomial formula ie ^0, ^1, ^2, ^3, ^4
    scan_power_gs = scan_g[...,na] ** scan_powers_s #raises scan_std to scan_powers_s ie x^0, x^1, x^2, x^3, x^4
    dstwtr_cal_gl0 = np.sum(uinp.sheep['i_scan_coeff_l0s'] * scan_power_gs[...,na,:], axis = -1) #add the coefficients and sum all the elements of the equation ie intercept+ax+bx^2+cx^3+dx^4

    ##convert the litter size proportion for the calibration period to the prediction period (the prediction period is the value of the cycles argument)
    dstwtr_gl0 = np.zeros_like(dstwtr_cal_gl0)
    dry_propn_cal_gl0 = dstwtr_cal_gl0[..., 0:1]
    dry_propn_gl0 = dry_propn_cal_gl0 ** (cycles / calibration_cycles)
    dstwtr_gl0[..., 0:1] = dry_propn_gl0
    dstwtr_gl0[..., 1:] = dstwtr_cal_gl0[..., 1:] * (1 - dry_propn_gl0) / (1 - dry_propn_cal_gl0)

    ##set values between 0 & 1 and adjust singles so that total is 1
    dstwtr_gl0 = np.clip(dstwtr_gl0, 0, 1)
    t_mask = [True, False, True, True]
    dstwtr_gl0[..., 1] = 1 - np.sum(dstwtr_gl0 * t_mask, axis = -1) # mask out the Singles value in the sum of the array across the l0 axis
    return dstwtr_gl0


def f1_btrt0(dstwtr_propn,lss,lstw,lstr): #^this function is inflexible ie if you want to add quadruplets
    '''
    Parameters
    ----------
    dstwtr_propn : np array, proportion of dams that are dry, singles, twin and triplets prior to birth.
    lss : np array, survival of single born progeny at birth.
    lstw : np array, survival of twin born progeny at birth.
    lstr : np array, survival of triplet born progeny at birth.

    Returns
    -------
    btrt_b0xyg : np array, proportion of lambs in each btrt category (eg 11, 22, 21 ...).
    progeny_total_xyg: np array, total number of progeny alive after birth per ewe mated

    '''
    ##progeny numbers is the number of alive progeny in each b0 slice per dam giving birth to that litter size
    ### value is the number of alive progeny in an outcome multiplied by the probability of the outcome.
    ### probability is based on survival of s, tw and tr at birth.
    progeny_numbers_b0yg = np.zeros((uinp.parameters['i_b0_len'],lss.shape[-2],lss.shape[-1]))
    progeny_numbers_b0yg[0,...] = lss
    progeny_numbers_b0yg[1,...] = 2 * lstw**2 #number of progeny surviving when there are no deaths is 2, therefore 2p^2
    progeny_numbers_b0yg[2,...] = 3 * lstr**3 #number of progeny surviving when there are no deaths is 3, therefore 3p^3
    progeny_numbers_b0yg[3,...] = 2 * lstw * (1 - lstw)  #the 2 is because it could be either progeny 1 that dies or progeny 2 that dies
    progeny_numbers_b0yg[4,...] = 2 * (3* lstr**2 * (1 - lstr))  #the 2x is because there are 2 progeny surviving in the litter and the 3x because it could be either progeny 1, 2 or 3 that dies
    progeny_numbers_b0yg[5,...] = 3* lstr * (1 - lstr)**2  #the 3x because it could be either progeny 1, 2 or 3 that survives
    ##mul progeny numbers array with number of dams giving birth to that litter size to get the number of progeny surviving per dam giving birth.
    a_nfoet_b0 = sinp.stock['a_nfoet_b1'][sinp.stock['i_mask_b0_b1']] #create association between l0 and b0
    btrt_b0yg = progeny_numbers_b0yg * dstwtr_propn[a_nfoet_b0]
    ##add singleton x axis
    btrt_b0xyg = np.expand_dims(btrt_b0yg, axis = tuple(range((uinp.parameters['i_cb0_pos'] + 1), -2))) #note i_cb0_pos refers to b0 position
    ##finally convert proportion from 'per dam' to 'per progeny'
    ###total number of progeny surviving per dam (similar to number of progeny marked)
    progeny_total_xyg = np.sum(btrt_b0xyg, axis=0)
    ###proportion of the progeny of each BTRT as a proportion of the progeny born alive
    btrt_propn_b0xyg = btrt_b0xyg / progeny_total_xyg
    return btrt_propn_b0xyg, progeny_total_xyg


#BTRT for b1 - code below not currently used but may be a little helpful later on.
# def f_btrt1(dstwtr,pss,pstw,pstr): #^this function is inflexible ie if you want to add quadruplets
#     '''
#     Parameters
#     ----------
#     dstwtr : np array
#         proportion of dry, singles, twin and triplets.
#     pss : np array
#         single survival.
#     pstw : np array
#         twin survival.
#     pstr : np array
#         triplet survival.

#     Returns
#     -------
#     btrt_b1nwzida0e0b0xyg : np array
#         probability of ewe with lambs in each btrt category (eg 11, 22, 21 ...).

#     '''

#     ##lamb numbers is the number of lambs in each b0 category, based on survival of s, tw and tr after birth.
#     lamb_numbers_b1yg = np.zeros((11,pss.shape[-2],pss.shape[-1])) #^where can i reference 11? would be good to have a b1 slice count somewhere.
#     lamb_numbers_b1yg[0,...] = pss
#     lamb_numbers_b1yg[1,...] = pstw**2
#     lamb_numbers_b1yg[2,...] = pstr**3
#     lamb_numbers_b1yg[3,...] = 2 * pstw * (1 - pstw)  #the 2 is because it could be either lamb 1 that dies or lamb 2 that dies
#     lamb_numbers_b1yg[4,...] = (3* pstr**2 * (1 - pstr))  # 3x because it could be either lamb 1, 2 or 3 that dies
#     lamb_numbers_b1yg[5,...] = 3* pstr * (1 - pstr)**2  #the 3x because it could be either lamb 1, 2 or 3 that survives
#     ##mul lamb numbers array with lambing percentage to get overall btrt
#     btrt_b1yg = lamb_numbers_b1yg * dstwtr_l0yg[sinp.stock['a_nfoet_b1'] ]
#     ##add singleton x axis
#     btrt_b1nwzida0e0b0xyg = np.expand_dims(btrt_b1yg, axis = tuple(range((uinp.parameters['i_cl1_pos'] + 1), -2))) #note i_cl1_pos refers to b1 position
#     return btrt_b1nwzida0e0b0xyg



################
#Sim functions #
################

def f1_feedsupply_adjust(attempts,feedsupply,itn):
    ##create empty array to put new feedsupply into, this is done so it doesnt have the itn axis (probably could just create from attempts array shape without last axis)
    feedsupply = np.zeros_like(feedsupply)
    ##which feedsupplies can be calculated using binary method - must have a negative and positive error
    binary_mask = np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1) < 0 #axis -1 is the itn axis ie take the min and max error from the previous iterations
    ##calc new feedsupply binary - take half of the two feedsupplies that have resulted in the error closest to 0. Only adds the binary result to slices that have a negative and a positive value (done using the mask created above)
    ###feedsupply with negative error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]>0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    neg_bool=np.ma.getdata(mask_attempts.max(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negative but closest to 0
    neg_bool = neg_bool * binary_mask[...,na] #this just makes sure the neg mask only has a true in the same slice as the pos array (so it can be applied to the feed supply array below)
    ###feedsupply with positive error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]<0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    pos_bool=np.ma.getdata(mask_attempts.min(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negative but closest to 0
    pos_bool = pos_bool * binary_mask[...,na] #this just makes sure the pos mask only has a true in the same place as the neg mask. 
    ##calc feedsupply
    feedsupply[binary_mask] = (attempts[...,0][neg_bool] + attempts[...,0][pos_bool])/2    
    ##calc feedsupply using interpolation
    ###first determine the slope, slope is always positive ie as feedsupply increases error increase because error = lwc - target and more feed means higher lwc.
    if itn==0:
        slope=pinp.sheep['i_feedsupply_slope_std']
    else:
        ####linregress only works on 1d array and can't use apply_over_axis because needs x and y. maybe there is a better way but i looked for a while and found nothing
        slope=np.empty_like(feedsupply)
        feedsupply_all_itn = attempts[...,0]
        error_all_itn = attempts[...,1]
        for i in np.ndindex(error_all_itn.shape[:-1]): #not exactly sure how this is working but it is creating tuple of each combo of slices in each axis.
            x= feedsupply_all_itn[i] #indexing with tuple works correctly if we are interested in the last axis otherwise it doesn't work properly for some reason.ie t[(0,0)] == t[0,0,:] but t[:,(0,0)] != t[:,0,0]
            y= error_all_itn[i]
            slope[i] = stats.linregress(x,y)
    ####new feedsupply = minerror / slope. It is assumed that the most recent itn has the most accurate feedsupply
    feedsupply[~binary_mask] = ((2 * attempts[...,-1,1]) / slope)[~binary_mask] # x2 to overshoot then switch to binary.
    return feedsupply


def f1_rev_update(trait_name, trait_value, rev_trait_value):
    trait_idx = sinp.structuralsa['rev_trait_name'].tolist().index(trait_name)
    if sinp.structuralsa['rev_trait_inc'][trait_idx]:
        if sinp.structuralsa['rev_create']:
            rev_trait_value[trait_name] = trait_value.copy() #have to copy so that traits (eg mort) that are added to using += do not also update the rev value
        else:
            trait_value = rev_trait_value[trait_name]
    return trait_value


def f1_history(history, new_value, days_in_period):
    '''
    The idea that the f1_history is implementing is for traits that have a lag from increased nutrition to increased production.
    The representation being that the production today is an average of the non-lagged estimated production from the last x days (where x is either len_p2 or len_p3, either of which can be 1 but are expected to be 25).
    The history function is keeping track of the last x days of estimated production (slice 0 is the most recent day, slice -1 is the oldest day of the history). 
    The process is:
        1. move the most recent history back (to make space for the production from this period: days_period) 
        2. make the 0:days_period = this period estimated production
    '''
    offset = np.max(np.minimum(days_in_period.astype(int), history.shape[0]))
    history = np.roll(history, offset, axis=0)
    ##make new value = nan when days per period==0
    days_in_period = np.broadcast_to(days_in_period, new_value.shape) #broadcast days array so it can be used to make mask
    new_value[days_in_period.astype(int)==0]=np.nan
    history[:offset, ...] = new_value
    weights = history!=np.nan
    t_history = np.nan_to_num(history) #convert nan to 0
    lagged = fun.f_weighted_average(t_history, weights=weights, axis = 0)
    return lagged, history


def f_potential_intake_cs(ci, cl, srw, relsize_start, rc_start, temp_lc_dams, temp_ave, temp_max, temp_min, rain_intake
                          , rc_birth_start=1, pi_age_y=0, lb_start=0, mp2=0, piyf=1, period_between_birthwean=1, sam_pi=1):
    '''

    :param ci:
    :param cl:
    :param srw:
    :param relsize_start:
    :param rc_start:
    :param temp_lc_dams:
    :param temp_ave:
    :param temp_max:
    :param temp_min:
    :param rain_intake:
    :param rc_birth_start:
    :param pi_age_y:
    :param lb_start:
    :param mp2:
    :param piyf:
    :param period_between_birthwean:
    :param sam_pi:
    :return:
    '''
    ##Condition factor on PI
    picf= np.minimum(1, rc_start * (ci[20, ...] - rc_start) / (ci[20, ...] - 1))
    ##Lactation adjustment (RC at parturition) - only active for dams
    la = 1 + ci[15, ...] * (rc_birth_start - 1)
    ##Lactation factor on PI - dam only
    pilf = 1 + pi_age_y * la * lb_start
    ##Temperature function
    piax = np.arccos(np.clip((temp_ave - temp_lc_dams) / (0.5 * (temp_max - temp_min)),-1,1))
    ##Temperature below the lower critical temp
    tlow = piax * (temp_lc_dams - temp_ave) + 0.5 * np.sin(piax) * (temp_max - temp_min) / np.pi
    ##Temperature factor on PI - high temperatures
    pitf_high = 1 - ci[5, ...] * (temp_ave - ci[6, ...])
    ##Temperature factor on PI - low temperatures
    pitf_low = 1 + ci[17,...] * tlow * rain_intake
    ##Temperature factor on PI
    pitf = np.minimum(1, pitf_high) * np.maximum(1, pitf_low)
    ##Potential intake
    pi = ci[1, ...] * srw * relsize_start * (ci[2, ...] - relsize_start) * picf * pitf * pilf * sam_pi
    ##Potential intake of pasture - young at foot only
    pi = (pi - mp2 / cl[6, ...] * cl[25, ...]) * piyf
    ##Potential intake of pasture - young at foot only
    pi = pi * period_between_birthwean
    return np.maximum(0,pi)


def f_potential_intake_mu(srw):
    pi = 0.028 * srw
    return np.maximum(0,pi)


def f_intake(pi, ri, md_herb, feedsupply, intake_s, i_md_supp, mp2=0):
    ##Pasture intake
    intake_f = np.maximum(0, pi - intake_s) * ri * (feedsupply < 3)
    ##ME intake from forage	
    mei_forage = 0
    ##ME intake from supplement	
    mei_supp = intake_s * i_md_supp
    ##ME intake from herbage	
    mei_herb = intake_f * md_herb
    ##ME intake of solid food	
    mei_solid = mei_forage + mei_herb + mei_supp
    ##M/D of the diet (solids)	
    md_solid = fun.f_divide(mei_solid, intake_f + intake_s) #func to stop div/0 error if fs  = 3 (eg zero feed)
    ##ME intake total	
    mei = mei_solid + mp2
    ##Proportion of ME as milk	
    mei_propn_milk = fun.f_divide(mp2, mei) #func to stop div/0 error - if mei is 0 then so is numerator
    ##Proportion of ME as herbage	
    mei_propn_herb = fun.f_divide((mei_herb + mei_forage), mei) #func to stop div/0 error - if mei is 0 then so is numerator
    ##Proportion of ME as supp	
    mei_propn_supp = fun.f_divide(mei_supp, mei) #func to stop div/0 error - if mei is 0 then so is numerator
    return mei, mei_solid, intake_f, md_solid, mei_propn_milk, mei_propn_herb, mei_propn_supp


def f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb
         , kl = 0, mei_propn_milk = 0, lact_propn = 0):
    '''

    Parameters
    ----------
    ck : Numpy array
        sim parameters - efficiency of energy use.
    belowmaint : Numpy array of Boolean
        Is the animal class in energy deficit.
    km : Numpy array of float
        Efficiency of energy use for maintenance.
    kg_supp : Numpy array
        efficiency of supplement energy consumed used for LWG.
    kg_fodd : Numpy array
        efficiency of herbage energy consumed used for LWG.
    mei_propn_supp : Numpy array
        Proportion of energy consumed that was from supplement.
    mei_propn_herb : Numpy array
        Proportion of energy consumed that was from herbage.
    mei_propn_milk : Numpy array
        Proportion of energy consumed that was from milk.
    kl : Numpy array of float, Optional
        Efficiency of energy use for lactation. The default is 0.
    lact_propn : Numpy array, optional
        Proportion of the period that the dam is lactating. The default is 0.

    Returns
    -------
    kg - Efficiency of energy used for growth.

    '''
    # ##Set days_lact to numpy array if arg value is 0
    # lact_propn = np.asarray(lact_propn)
    ##Lactating, losing weight
    kg_lact_lose = kl / ck[10, ...]
    ##Lactation other
    kg_lact = kl * ck[9, ...]
    ##Non-lactating, losing weight
    kg_dry_lose = km / ck[11, ...]
    ##Non-lactating, other
    kg_dry = kg_supp * mei_propn_supp + kg_fodd * mei_propn_herb + ck[12, ...] * mei_propn_milk
    ##Compile lactation
    kg_lact = fun.f_update(kg_lact , kg_lact_lose , belowmaint)
    ##Compile non-lactating
    kg_dry = fun.f_update(kg_dry , kg_dry_lose , belowmaint)
    ##Compile whole formula
    kg = lact_propn * kg_lact + (1 - lact_propn) * kg_dry
    return kg


def f_energy_cs(ck, cx, cm, lw_start, ffcfw_start, mr_age, mei, omer_history_start, days_period, md_solid, i_md_supp,
                md_herb, lgf_eff, dlf_eff, i_steepness, density, foo, feedsupply, intake_f, dmd, mei_propn_milk=0, sam_kg=1, sam_mr=1):
    ##Efficiency for maintenance	
    km = (ck[1, ...] + ck[2, ...] * md_solid) * (1-mei_propn_milk) + ck[3, ...] * mei_propn_milk
    ##Efficiency for lactation - dam only	
    kl =  ck[5, ...] + ck[6, ...] * md_solid
    ##Efficiency for growth (supplement) including the sensitivity scalar
    kg_supp = ck[16, ...] * i_md_supp * sam_kg
    ##Efficiency for growth (fodder) including the sensitivity scalar
    kg_fodd = ck[13, ...] * lgf_eff * (1+ ck[15, ...] * dlf_eff) * md_herb * sam_kg
    ##Energy required at maint for metabolism	
    emetab = cx[10, ...] * cm[2, ...] * ffcfw_start ** 0.75 * mr_age * (1 + cm[5, ...] * mei_propn_milk)
    ##Distance walked (horizontal equivalent)	
    distance = (1 + np.tan(np.deg2rad(i_steepness))) * np.minimum(1, cm[17, ...] / density) / (cm[8, ...] * foo + cm[9, ...])
    ##Set Distance walked to 0 if in confinement	
    distance = distance * (feedsupply < 3)
    ##Energy required for movement	
    emove = cm[16, ...] * distance * lw_start
    ##Energy required for grazing (chewing and walking around)
    egraze = cm[6, ...] * ffcfw_start * intake_f * (cm[7, ...] - dmd) + emove
    ##Energy associated with organ activity
    omer, omer_history = f1_history(omer_history_start, cm[1, ...] * mei, days_period)
    ##ME requirement for maintenance (before ECold)
    meme = ((emetab + egraze) / km + omer) * sam_mr
    return meme, omer_history, km, kg_fodd, kg_supp, kl


def f_foetus_cs(cp, cb1, kc, nfoet, relsize_start, rc_start, w_b_std_y, w_f_start, nw_f_start, nwf_age_f, guw_age_f, dce_age_f):
    ##expected normal birth weight with dam age adj.
    w_b_exp_y = (1 - cp[4, ...] * (1 - relsize_start)) * w_b_std_y
    ##Normal weight of foetus (mid period - dam calcs)	
    nw_f = w_b_exp_y * nwf_age_f
    ##change in normal weight of foetus	
    d_nw_f = nw_f - nw_f_start
    ##Proportion of normal foetal and birth weights	
    nwf_nwb = fun.f_divide(nw_f, w_b_std_y)
    ##Normal weight of individual conceptus (mid period)	
    nw_gu = cp[5, ...] * w_b_exp_y * guw_age_f
    ##Normal energy of individual conceptus (end of period)	
    normale_dgu = cp[8, ...] * cp[5, ...] * w_b_exp_y * dce_age_f
    # normale_dgu = cp[8, ...] * cp[5, ...] * w_b_exp_y * ce_age_f
    ##Condition factor on BW
    cfpreg = (rc_start - 1) * nwf_nwb
    ##change in foetus weight	
    d_w_f = d_nw_f *(1 + np.minimum(cfpreg, cfpreg * cb1[14, ...]))
    ##foetus weight (end of period)	
    w_f = w_f_start + d_w_f
    ##Weight of the gravid uterus (conceptus - mid period)	
    guw = nfoet * (nw_gu + (w_f - nw_f))
    ##Body condition of the foetus	
    rc_f = fun.f_divide(w_f, nw_f) #func to handle div0 error
    ##Cumulative ME required for conceptus	
    # nec_cum = nfoet * rc_f * normale_dgu
    nec = nfoet * rc_f * normale_dgu
    ##NE required for conceptus
    # nec = np.maximum(0,fun.f_divide(nec_cum - nec_cum_start, days_period_f))
    ##ME required for conceptus	
    mec = nec / kc
    # return w_f, nec_cum, mec, nec, w_b_exp_y, nw_f, guw
    return w_f, mec, nec, w_b_exp_y, nw_f, guw


def f1_carryforward_u1(cu1, cg, ebg, period_between_joinstartend, period_between_mated90, period_between_d90birth
                      , period_between_birthwean, days_period, period_propn=1):
    ##Select coefficient to increment the carry forward quantity based on the current period
    ### can only be the coefficient from one of the periods and the later period overwrites the earlier period.
    coeff_cf1 = fun.f_update(0, cu1[1,...], period_between_joinstartend) #note cu1 has already had the first axis (production parameter) sliced when it was passed in
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[2,...], period_between_mated90)
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[3,...], period_between_d90birth)
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[4,...], period_between_birthwean)
    ##Calculate the increment (d_cf) from the coefficient, the change in LW (kg/d) and the days per period
    d_cf = coeff_cf1 * ebg * cg[18, ...] * days_period * period_propn
    return d_cf


def f_birthweight_cs(cx, w_b_yatf, w_f_dams, period_is_birth):
    ##set BW = foetal weight at end of period (if born)	
    t_w_b = w_f_dams * cx[15, ...] * period_is_birth
    ##update birth weight if it is birth period
    w_b_yatf = fun.f_update(w_b_yatf, t_w_b, period_is_birth)
    return w_b_yatf


def f_birthweight_mu(cu1, cb1, cg, cx, ce, w_b, cf_w_b_dams, ffcfw_birth_dams, ebg_dams, days_period, gest_propn
                     , period_between_joinscan, period_between_scanbirth, period_is_birth):
    ##Carry forward BW increment	
    d_cf_w_b = f1_carryforward_u1(cu1[16, ...], cg, ebg_dams, False, period_between_joinscan, period_between_scanbirth
                                 , False, days_period, gest_propn)
    ##Increment the total carry forward BW
    cf_w_b_dams = cf_w_b_dams + d_cf_w_b
    ##estimate BW by including the intercept, the effect of dam weight at birth and other non-LW coefficients
    t_w_b_yatf = (cf_w_b_dams + cu1[16, -1, ...] + cu1[16, 0, ...] * ffcfw_birth_dams + cb1[16, ...]
                  + cx[16, ...] + ce[16, ...])
    ##Update w_b if period is birth
    w_b = fun.f_update(w_b, t_w_b_yatf, period_is_birth)
    return w_b, cf_w_b_dams


def f_weanweight_cs(w_w_yatf, ffcfw_start_yatf, ebg_yatf, days_period, period_is_wean):
    ##set WWt = yatf weight at weaning	
    t_w_w = (ffcfw_start_yatf + ebg_yatf * days_period)
    ##update weaning weight if it is weaning period
    w_w_yatf = fun.f_update(w_w_yatf, t_w_w, period_is_wean)
    return w_w_yatf


def f_weanweight_mu(cu1, cb1, cg, cx, ce, nyatf, w_w, cf_w_w_dams, ffcfw_wean_dams, ebg_dams, foo, foo_ave_start
                    , days_period, day_of_lactation, period_between_joinscan, period_between_scanbirth
                    , period_between_birthwean, period_is_wean):
    ##Calculate average FOO to end of this period (increment the running average to date)
    foo_ave_end = fun.f_divide(foo_ave_start * day_of_lactation + foo * days_period, day_of_lactation + days_period)
    ##Carry forward WWt increment
    d_cf_w_w = f1_carryforward_u1(cu1[17, ...], cg, ebg_dams, False, period_between_joinscan, period_between_scanbirth
                                 , period_between_birthwean, days_period)
    ##Increment the total Carry forward WWt
    cf_w_w_dams = cf_w_w_dams + d_cf_w_w
    ##add intercept, impact of dam LW at weaning, FOO, BTRT, gender and dam age effects to the carry forward value
    t_w_w = (cf_w_w_dams + cu1[17, -1, ...] + cu1[17, 0, ...] * ffcfw_wean_dams + cu1[17, 5, ...] * foo_ave_end
             + cu1[17, 6, ...] * foo_ave_end ** 2 + cb1[17, ...] + cx[17, ...] + ce[17, ...]) * (nyatf > 0)
    ##Update w_w if it is weaning	
    w_w = fun.f_update(w_w, t_w_w, period_is_wean)
    return w_w, cf_w_w_dams, foo_ave_end


#todo Consider combining into 1 function f_progenyflc_mu
def f_progenycfw_mu(cu1, cg, cfw_adj, cf_cfw_dams, ffcfw_birth_dams, ffcfw_birth_std_dams, ebg_dams, days_period
                    , gest_propn, period_between_mated90, period_between_d90birth, period_is_birth):
    ##impact on progeny CFW of the dam LW profile being different from the standard pattern
    ### LTW coefficients are multiplied by the difference in the LW profile from the standard profile. This only requires representing explicitly for LW at birth because the std LW change is 0. Std pattern is lambing in CS 3, so LW = normal weight
    ##Carry forward CFW increment
    d_cf_cfw = f1_carryforward_u1(cu1[12, ...], cg, ebg_dams, False, period_between_mated90, period_between_d90birth, False, days_period, gest_propn)
    ##Increment the total Carry forward CFW
    cf_cfw_dams = cf_cfw_dams + d_cf_cfw
    ##temporary calculation including difference in current dam LW (only used if period is birth)
    ### Birth coefficient multiplied by the difference from the standard pattern rather than absolute weight
    t_cfw_yatf = (cf_cfw_dams + cu1[12, -1, ...] + cu1[12, 0, ...] * (ffcfw_birth_dams - ffcfw_birth_std_dams))
    ##Update CFW if it is birth
    cfw_adj = fun.f_update(cfw_adj, t_cfw_yatf, period_is_birth)
    return cfw_adj, cf_cfw_dams


def f_progenyfd_mu(cu1, cg, fd_adj, cf_fd_dams, ffcfw_birth_dams, ffcfw_birth_std_dams, ebg_dams, days_period
                   , gest_propn, period_between_mated90, period_between_d90birth, period_is_birth):
    ##impact on progeny FD of the dam LW profile being different from the standard pattern
    ### LTW coefficients are multiplied by the difference in the LW profile from the standard profile. This only requires representing explicitly for LW at birth because the std LW change is 0. Std pattern is lambing in CS 3, so LW = normal weight
    ##Carry forward FD increment
    d_cf_fd = f1_carryforward_u1(cu1[13, ...], cg, ebg_dams, False, period_between_mated90, period_between_d90birth, False, days_period, gest_propn)
    ##Increment the total Carry forward FD
    cf_fd_dams = cf_fd_dams + d_cf_fd
    ##temporary calculation including difference in current dam LW (only used if period is birth)
    ### Birth coefficient multiplied by the difference from the standard pattern rather than absolute weight
    t_fd_yatf = (cf_fd_dams + cu1[13, -1, ...] + cu1[13, 0, ...] * (ffcfw_birth_dams - ffcfw_birth_std_dams))
    ##Update FD if it is birth
    fd_adj = fun.f_update(fd_adj, t_fd_yatf, period_is_birth)
    return fd_adj, cf_fd_dams


def f_milk(cl, srw, relsize_start, rc_birth_start, mei, meme, mew_min, rc_start, ffcfw75_exp_yatf, lb_start, ldr_start, age_yatf, mp_age_y,  mp2_age_y, i_x_pos, days_period_yatf, kl, lact_nut_effect):
    ##Max milk prodn based on dam rc birth
    mpmax = srw** 0.75 * relsize_start * rc_birth_start * lb_start * mp_age_y
    ##Excess ME available for milk	
    mel_xs = np.maximum(0, (mei - (meme + mew_min * relsize_start))) * cl[5, ...] * kl
    ##Excess ME as a ratio of mpmax
    milk_ratio = fun.f_divide(mel_xs, mpmax) #func stops div0 error - and milk ratio is later discarded because days period f = 0
    ##Age or energy factor
    ad = np.maximum(age_yatf, milk_ratio / (2 * cl[22, ...]))
    ##Milk production based on energy available	
    mp1 = cl[7, ...] * mpmax / (1 + np.exp(-(-cl[19, ...] + cl[20, ...] * milk_ratio + cl[21, ...] * ad * (milk_ratio - cl[22, ...] * ad) - cl[23, ...] * rc_start * (milk_ratio - cl[24, ...] * rc_start))))
    ##Milk production (per animal) based on suckling volume	(milk production per day of lactation)
    mp2 = np.minimum(mp1, np.mean(fun.f_dynamic_slice(ffcfw75_exp_yatf, i_x_pos, 1, None), axis = i_x_pos, keepdims=True) * mp2_age_y)   # averages female and castrates weight, ffcfw75 is metabolic weight
    ##ME for lactation (per day lactating)	
    mel = mp2 / (cl[5, ...] * kl)
    ##NE for lactation	
    nel = kl * mel
    ##ratio of actual to potential milk	
    dr = fun.f_divide(mp2, mpmax) #div func stops div0 error - and milk ratio is later discarded because days period f = 0
    ##Lagged DR (lactation deficit)
    ldr = (ldr_start - dr) * (1 - cl[18, ...]) ** days_period_yatf + dr
    ##Loss of potential milk due to consistent under production	
    lb = lb_start - cl[17, ...] / cl[18, ...] * (1 - cl[18, ...]) * (1 - (1 - cl[18, ...]) ** days_period_yatf) * (ldr_start - dr)
    ##If early in lactation = 1	
    lb = lb * lact_nut_effect + ~lact_nut_effect
    return mp2, mel, nel, ldr, lb


def f_fibre(cw_g, cc_g, ffcfw_start_g, relsize_start_g, d_cfw_history_start_p2g, mei_g, mew_min_g, d_cfw_ave_g
            , sfd_a0e0b0xyg, wge_a0e0b0xyg, af_wool_g, dlf_wool_g,  kw_yg, days_period_g, sfw_ltwadj_g, sfd_ltwadj_g
            , rev_trait_value, mec_g1=0, mel_g1=0, gest_propn_g1=0, lact_propn_g1=0, sam_pi=1):
    ##adjust wge, cfw_ave, mew_min & sfd for the LTW adjustments (CFW is a scalar and FD is an addition)
    wge_a0e0b0xyg = wge_a0e0b0xyg * sfw_ltwadj_g
    d_cfw_ave_g = d_cfw_ave_g * sfw_ltwadj_g
    mew_min_g = mew_min_g * sfw_ltwadj_g
    sfd_a0e0b0xyg = sfd_a0e0b0xyg + sfd_ltwadj_g
    ##adjust wge by sam_pi so the intake sensitivity doesn't alter the wool growth outcome for the genotype
    ###this scaling could be applied to sfw but is applied here so that pi can be altered for a single age group
    ###which is required for the GEPEP analysis that is calibrating the adult intake and the fleece weight
    wge_a0e0b0xyg = wge_a0e0b0xyg / sam_pi
    ##ME available for wool growth
    mew_xs_g = np.maximum(mew_min_g * relsize_start_g, mei_g - (mec_g1 * gest_propn_g1 + mel_g1 * lact_propn_g1))
    ##Wool growth (protein weight-as shorn i.e. not DM) if there was no lag
    d_cfw_nolag_g = cw_g[8, ...] * wge_a0e0b0xyg * af_wool_g * dlf_wool_g * mew_xs_g
    ##Process the CFW REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    d_cfw_nolag_g = f1_rev_update('cfw', d_cfw_nolag_g, rev_trait_value)
    ##Wool growth (protein weight) with lag and updated history
    d_cfw_g, d_cfw_history_p2g = f1_history(d_cfw_history_start_p2g, d_cfw_nolag_g, days_period_g)
    ##Net energy required for wool
    new_g = cw_g[1, ...] * (d_cfw_g - cw_g[2, ...] * relsize_start_g) / cw_g[3, ...]
    ##ME required for wool (above basal growth rate)
    mew_g = new_g / kw_yg #can be negative because mem assumes 4g of wool is grown. If less is grown then mew 'returns' the energy.
    ##Fibre diameter for the days growth
    d_fd_g = sfd_a0e0b0xyg * fun.f_divide(d_cfw_g, d_cfw_ave_g) ** cw_g[13, ...]  #func to stop div/0 error when d_cfw_ave=0 so does d_cfw (only have a 0 when day period = 0)
    ##Process the FD REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    d_fd_g = f1_rev_update('fd', d_fd_g, rev_trait_value)
    ##Surface Area
    area = cc_g[1, ...] * ffcfw_start_g ** (2/3)
    ##Daily fibre length growth
    d_fl_g = 100 * fun.f_divide(d_cfw_g, cw_g[10, ...] * cw_g[11, ...] * area * np.pi * (0.5 * d_fd_g / 10**6) ** 2) #func to stop div/0 error, when d_fd==0 so does d_cfw
    return d_cfw_g, d_fd_g, d_fl_g, d_cfw_history_p2g, mew_g, new_g


def f_chill_cs(cc, ck, ffcfw_start, rc_start, sl_start, mei, meme, mew, new, km, kg_supp, kg_fodd, mei_propn_supp
               , mei_propn_herb, temp_ave_a1e1b1nwzida0e0b0xyg, temp_max_a1e1b1nwzida0e0b0xyg, temp_min_a1e1b1nwzida0e0b0xyg
               , ws_a1e1b1nwzida0e0b0xyg, rain_a1e1b1nwzida0e0b0xygp1, index_m0, guw = 0, kl = 0, mei_propn_milk = 0
               , mec = 0, mel = 0, nec = 0, nel = 0, gest_propn	= 0, lact_propn = 0):
    ##Animal is below maintenance
    belowmaint = mei < (meme + mec + mel + mew)
    ##Efficiency for growth (before ECold)
    kge = f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
    ##Sinusoidal variation in temp & wind
    sin_var_m0 = np.sin(2 * np.pi / 12 *(index_m0 - 3))
    ##Ambient temp (2 hourly)
    temperature_a1e1b1nwzida0e0b0xygm0 = temp_ave_a1e1b1nwzida0e0b0xyg[..., na] + (temp_max_a1e1b1nwzida0e0b0xyg[..., na] - temp_min_a1e1b1nwzida0e0b0xyg[..., na]) / 2 * sin_var_m0
    ##Wind velocity (2 hourly)
    wind_a1e1b1nwzida0e0b0xygm0 = ws_a1e1b1nwzida0e0b0xyg[..., na] * (1 + 0.35 * sin_var_m0)
    ##Proportion of sky that is clear
    sky_clear_a1e1b1nwzida0e0b0xygp1 = 0.7 * np.exp(-0.25 * rain_a1e1b1nwzida0e0b0xygp1)
    ##radius of animal
    radius = np.maximum(0.001,cc[2, ...] * ffcfw_start ** (1/3)) #max because realistic values of radius can be small for lambs - stops div0 error
    ##surface area of animal
    area = np.maximum(0.001,cc[1, ...] * ffcfw_start ** (2/3)) #max because area is in m2 so realistic values of area can be small for lambs
    ##Impact of wet fleece on insulation
    wetflc_a1e1b1nwzida0e0b0xygp1 = cc[5, ..., na] + (1 - cc[5, ..., na]) * np.exp(-cc[6, ..., na] * rain_a1e1b1nwzida0e0b0xygp1 / sl_start[..., na])
    ##Insulation of air (2 hourly)
    in_air_a1e1b1nwzida0e0b0xygm0 = radius[..., na] / (radius[..., na] + sl_start[..., na]) / (cc[7, ..., na] + cc[8, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of coat (2 hourly)
    in_coat_a1e1b1nwzida0e0b0xygm0 = radius[..., na] * np.log((radius[..., na] + sl_start[..., na]) / radius[..., na]) / (cc[9, ..., na] - cc[10, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of  tissue
    in_tissue = cc[3, ...] * (rc_start - cc[4, ...] * (rc_start - 1))
    ##Insulation of  air + coat (2 hourly)
    in_ext_a1e1b1nwzida0e0b0xygm0p1 = wetflc_a1e1b1nwzida0e0b0xygp1[..., na, :] * (in_air_a1e1b1nwzida0e0b0xygm0[..., na] + in_coat_a1e1b1nwzida0e0b0xygm0[..., na])
    ##Impact of clear night skies on ME loss
    sky_temp_a1e1b1nwzida0e0b0xygm0p1 = sky_clear_a1e1b1nwzida0e0b0xygp1[..., na, :] * cc[13,..., na, na] * np.exp(-cc[14, ..., na, na] * np.minimum(0, cc[15, ..., na, na] - temperature_a1e1b1nwzida0e0b0xygm0[..., na]) ** 2)
    ##Heat production per m2
    heat = (mei - nec * gest_propn - nel * lact_propn - new - kge * (mei
            - (meme + mec * gest_propn + mel * lact_propn + mew))
            + cc[16, ...] * guw) / area
    ##Lower critical temperature (2 hourly)
    temp_lc_a1e1b1nwzida0e0b0xygm0p1 = cc[11, ..., na, na]+ cc[12, ..., na, na] - heat[..., na, na] * (in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0p1) + sky_temp_a1e1b1nwzida0e0b0xygm0p1
    ##Lower critical temperature (period)
    temp_lc_a1e1b1nwzida0e0b0xyg = np.average(temp_lc_a1e1b1nwzida0e0b0xygm0p1, axis = (-1,-2))
    ##Extra ME required to keep warm
    mecold_a1e1b1nwzida0e0b0xyg = area * np.average(fun.f_dim(temp_lc_a1e1b1nwzida0e0b0xygm0p1, temperature_a1e1b1nwzida0e0b0xygm0[..., na]) /(in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0p1), axis = (-1,-2))
    ##ME requirement for maintenance (inc ECold)
    mem = meme + mecold_a1e1b1nwzida0e0b0xyg
    ##Animal is below maintenance (incl ecold)
    belowmaint = mei < (mem + mec + mel + mew)
    ##Efficiency for growth (inc ECold) -different to the second line because belowmaint includes ecold
    kg = f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
    return mem, temp_lc_a1e1b1nwzida0e0b0xyg, kg


def f_lwc_cs(cg, rc_start, mei, mem, mew, z1f, z2f, kg, rev_trait_value, mec = 0, mel = 0, gest_propn = 0, lact_propn = 0):
    ## requirement for maintenance
    maintenance = mem + mec * gest_propn + mel * lact_propn + mew
    ##Level of feeding (maint = 0)
    level = (mei / maintenance) - 1
    ##Energy intake that is surplus to maintenance
    surplus_energy = mei - maintenance
    ##Net energy gain (based on ME)
    neg = kg * surplus_energy
    ##Energy Value of gain
    evg = cg[8, ...] - z1f * (cg[9, ...] - cg[10, ...] * (level - 1)) + z2f * cg[11, ...] * (rc_start - 1)
    ##Protein content of gain (some uncertainty for sign associated with zf2.
    ### GrazFeed documentation had +ve however, this implies that PCG increases when BC > 1. So changed to -ve
    #todo check this equation when converting to a heat production based model.
    pcg = cg[12, ...] + z1f * (cg[13, ...] - cg[14, ...] * (level - 1)) - z2f * cg[15, ...] * (rc_start - 1)
    ##Empty bodyweight gain
    ebg = neg / evg
    ##Process the Liveweight REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    ebg = f1_rev_update('lwc', ebg, rev_trait_value)
    ##Protein gain
    pg = pcg * ebg
    ##fat gain
    fg = (neg - pg * cg[21, ...]) / cg[22, ...]
    return ebg, evg, pg, fg, level, surplus_energy


def f_lwc_mu(cg, rc_start, mei, mem, mew, z1f, z2f, kg, rev_trait_value, mec = 0, mel = 0, gest_propn = 0, lact_propn = 0):
    ## requirement for maintenance
    maintenance = mem + mec * gest_propn + mel * lact_propn + mew
    ##Level of feeding (maint = 0)
    level = (mei / maintenance) - 1
    ##Energy intake that is surplus to maintenance
    surplus_energy = mei - maintenance
    ##Net energy gain (based on ME)
    neg = kg * surplus_energy
    ##Energy Value of gain as calculated.
    c_evg = cg[8, ...] - z1f * (cg[9, ...] - cg[10, ...] * (level - 1)) + z2f * cg[11, ...] * (rc_start - 1)
    # evg = fun.f_update(evg , temporary, z2f < 1)
    ## Scale from calculated to input evg based on z2f. If z2f = 1 then use the value from the GEPEP trial
    evg = c_evg * (1 + sen.sap['evg'] * z2f)
    ##Empty bodyweight gain
    ebg = neg / evg
    ##Process the Liveweight REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    ebg = f1_rev_update('lwc', ebg, rev_trait_value)
    # ##Protein gain
    # pg = pcg * ebg
    # ##fat gain
    # fg = (neg - pg * cg[21, ...]) / cg[22, ...]
    ## proportion of fat and lean is determined from the EVG based on energy and DM content of muscle and adipose
    adipose_propn = (evg - (cg[21, ...] * cg[19, ...])) / ((cg[22, ...] * cg[20, ...]) - (cg[21, ...] * cg[19, ...]))
    fg = ebg * adipose_propn * cg[20, ...]
    pg = (neg - fg * cg[22, ...]) / cg[21, ...]
    return ebg, evg, pg, fg, level, surplus_energy


def f_wbe(aw, mw, cg):
    ## calculate whole body energy content from weight of adipose tissue (aw) and muscle (mw), and the dry matter content and energy density.
    wbe = aw * cg[20, ...] * cg[22, ...] + mw * cg[19, ...] * cg[21, ...]
    return wbe


def f_emissions_bc(ch, intake_f, intake_s, md_solid, level):
    ##Methane production total
    ch4_total = ch[1, ...] * (intake_f + intake_s)*((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))
    ##Methane production animal component
    ch4_animal = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)
    return ch4_total, ch4_animal


def f1_convert_fs2nv(fs_input, nv_p6f, feedsupply_f, a_p6_pa1e1b1nwzida0e0b0xyg):
    ##convert a feed supply array (feed) and return an NV array using a conversion array (nv_p6f) for a corresponding feedsupply (feedsupply_f)
    ## expect feed to have a p axis as axis 0.
    ###the position of the feedsupply input in the conversion array
    fs_col_pa1e1b1nwzida0e0b0xyg = np.searchsorted(feedsupply_f, fs_input, 'right') - 1
    fs_col_pa1e1b1nwzida0e0b0xyg = np.maximum(0, fs_col_pa1e1b1nwzida0e0b0xyg)
    ###the value from the conversion array in column fs_col in the row associated with the feed period for that generator period.
    nv_pa1e1b1nwzida0e0b0xygf = nv_p6f[a_p6_pa1e1b1nwzida0e0b0xyg, :]
    nv_pa1e1b1nwzida0e0b0xygf = np.take_along_axis(nv_pa1e1b1nwzida0e0b0xygf, fs_col_pa1e1b1nwzida0e0b0xyg[...,na], axis=-1)
    return nv_pa1e1b1nwzida0e0b0xygf[...,0] #remove singleton f axis - no longer needed.


def f1_convert_nv2fs(nv_input, nv_p6f, feedsupply_f, a_p6_pz):
    ##convert a feed supply array (feed) and return an NV array using a conversion array (nv_p6f) for a corresponding feedsupply (feedsupply_f)
    ## expect feed to have a p axis as axis 0.
    ### multi dim search sorted requires the axes to be the same, so convert p6 to p in the lookup array
    nv_pzf = nv_p6f[a_p6_pz, :]
    ###the position of the feedsupply input in the conversion array
    z_pos = sinp.stock['i_z_pos']
    fs_col_pa1e1b1nwzida0e0b0xyg = fun.searchsort_multiple_dim(nv_pzf, nv_input, axis_a0=0, axis_v0=0, axis_a1=1, axis_v1=z_pos, side='right') - 1
    fs_col_pa1e1b1nwzida0e0b0xyg = np.maximum(0, fs_col_pa1e1b1nwzida0e0b0xyg)
    ###the value from the feedsupply array in column fs_col.
    fs = feedsupply_f[fs_col_pa1e1b1nwzida0e0b0xyg]
    return fs


def f1_feedsupply(feedsupply_std_a1e1b1nwzida0e0b0xyg, paststd_foo_a1e1b1j0wzida0e0b0xyg, paststd_dmd_a1e1b1j0wzida0e0b0xyg
                 , paststd_hf_a1e1b1j0wzida0e0b0xyg, pi):
    ##level of pasture
    level_a1e1b1nwzida0e0b0xyg = np.trunc(np.minimum(2, feedsupply_std_a1e1b1nwzida0e0b0xyg)).astype('int') #note np.trunc rounds down to the nearest int (need to specify int type for the take along axis function below)
    ##next level up of pasture
    next_level_a1e1b1nwzida0e0b0xyg = np.minimum(2, level_a1e1b1nwzida0e0b0xyg + 1)
    ##decimal component of feedsupply
    proportion_a1e1b1nwzida0e0b0xyg = feedsupply_std_a1e1b1nwzida0e0b0xyg % 1
    ##foo (corrected for measurement system of the region and the pasture stage)
    paststd_foo_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    paststd_foo_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_a1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    foo_a1e1b1nwzida0e0b0xyg = paststd_foo_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_foo_next_a1e1b1nwzida0e0b0xyg - paststd_foo_a1e1b1nwzida0e0b0xyg)
    ##dmd
    paststd_dmd_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    paststd_dmd_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_a1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    dmd_a1e1b1nwzida0e0b0xyg = paststd_dmd_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_dmd_next_a1e1b1nwzida0e0b0xyg - paststd_dmd_a1e1b1nwzida0e0b0xyg)
    ##hf
    paststd_hf_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_hf_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    paststd_hf_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_hf_a1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, sinp.stock['i_n_pos'])
    hf_a1e1b1nwzida0e0b0xyg = paststd_hf_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_hf_next_a1e1b1nwzida0e0b0xyg - paststd_hf_a1e1b1nwzida0e0b0xyg)
    ##proportion of PI that is offered as supp
    supp_propn_a1e1b1nwzida0e0b0xyg = proportion_a1e1b1nwzida0e0b0xyg * (feedsupply_std_a1e1b1nwzida0e0b0xyg > 2) + (feedsupply_std_a1e1b1nwzida0e0b0xyg == 4)   # the proportion of diet if the value is above 2 and equal to 1.0 if fs==4 (at fs 3 sheep have 0 sup and 0 fodder at fs4 sheep have 100% of pi is sup)
    intake_s = pi * supp_propn_a1e1b1nwzida0e0b0xyg
    ##calc herb md
    herb_md = fsfun.dmd_to_md(dmd_a1e1b1nwzida0e0b0xyg)
    return foo_a1e1b1nwzida0e0b0xyg, hf_a1e1b1nwzida0e0b0xyg, dmd_a1e1b1nwzida0e0b0xyg, intake_s, herb_md


def f_conception_cs(cf, cb1, relsize_mating, rc_mating, crg_doy, nfoet_b1any, nyatf_b1any, period_is_mating, index_e1
                    , rev_trait_value, saa_rr):
    ''''
    Calculation of dam conception using CSIRO equation system

    The general approach is to calculate the probability of conception greater than or equal to 1,2,3 foetuses
    Probability is calculated from a sigmoid relationship based on relative size * relative condition at joining
    The estimation of cumulative probability is scaled by a factor that varies with (litter size * latitude * day of the year)
    The probability is an estimate of the number of dams carrying that number in the third trimester.
    Some dams conceive (and don't return to service) but don't carry to the third trimester, this is taken into account.
    The values are altered by a sensitivity analysis on scanning percentage
    Conception (proportion of dams that are dry) and litter size (number of foetuses per pregnant dam) can be controlled for relative economic values

    :param cf:
    :param cb1: GrazPlan parameter stating the probability of conception with different number of foetuses.
    :param relsize_mating:
    :param rc_mating:
    :param crg_doy:
    :param nfoet_b1any:
    :param nyatf_b1any:
    :param period_is_mating:
    :param index_e1:
    :param rev_trait_value:
    :param saa_rr:
    :return: Dam conception.
    '''
    if ~np.any(period_is_mating):
        conception = np.zeros_like(relsize_mating)
    else:
        b1_pos = sinp.stock['i_b1_pos']  #because used in many places in the function

        ## relative size and relative condition of the dams at mating are the determinants of conception
        ### the dams being mated are those in slices e1[0] and b1[0] (first cycle, not mated)
        relsize_mating_e1b1sliced = fun.f_dynamic_slice(relsize_mating, sinp.stock['i_e1_pos'], 0, 1, b1_pos, 0, 1) #take slice from e1 & b1 axis
        rc_mating_e1b1sliced = fun.f_dynamic_slice(rc_mating, sinp.stock['i_e1_pos'], 0, 1, b1_pos, 0, 1) #take slice from e1 & b1 axis

        ## probability of at least a given number of foetuses
        crg = crg_doy * fun.f_sig(relsize_mating_e1b1sliced * rc_mating_e1b1sliced, cb1[2, ...], cb1[3, ...])
        ##Set proportions to 0 for dams that gave birth and lost - this is required so that numbers in pp calculate correctly
        crg *= (nfoet_b1any == nyatf_b1any)

        ##Temporary array for probability of a given number of foetuses (calculated from the difference in the cumulative probability)
        ##Calculate probability from cumulative probability by the difference between the array and the array values offset by one slice (difference between '>x' and '>x+1').
        ### Define the temp array shape & populate with values from crg (values are required for the proportion of the highest parity dams)
        t_cr = crg.copy()
        slc = [slice(None)] * len(t_cr.shape)
        slc[b1_pos] = slice(1,-1)
        t_cr[tuple(slc)] = np.maximum(0, fun.f_dynamic_slice(crg, b1_pos, 1, -1) - fun.f_dynamic_slice(crg, b1_pos, 2, None))

        ##Apply scanning percentage sa to adjust the probability of the number of foetuses.
        ### Carried out here so that the sa affects the REV and is included in proportion of NM
        ### Achieved by calculating the impact of the sa on the scanning percentage and the change in the 'standardised'
        ### proportions of DST. Then adjusting the actual proportions of dry, singles and twins by that amount.
        # Calculate the repro rate from the probabilities above (t_cr) and convert to an expected proportion of dry,
        # # singles, twins & triplets after 1 cycle
        #### convert the proportion of DST in t_cr to an equivalent scanning % after the calibration number of cycles.
        repro_rate = f1_convert_scancycles(t_cr, nfoet_b1any, cycles = 1)
        ####remove singleton b1 axis by squeezing because it is replaced by the l0 axis in f1_DSTw
        repro_rate = np.squeeze(repro_rate, axis=b1_pos)
        saa_rr = np.squeeze(saa_rr, axis=b1_pos)
        ## apply the sa to the repro rate and convert the adjusted value to a proportion of dry, singles, twins & triplets after 1 cycle
        repro_rate_adj = fun.f_sa(repro_rate, sen.sam['rr'])
        repro_rate_adj = fun.f_sa(repro_rate_adj, saa_rr, 2) * (repro_rate > 0)     # only non-zero if original value was non-zero
        #### Convert the repro rate and adjusted repro rate to a 'standardised' proportion of DST after 1 cycle
        #### The proportions returned are in axis -1 and needs the slices altered (shape of l0 to b1) and moving to b1 position.
        propn_dst = np.moveaxis(f1_DSTw(repro_rate, cycles=1)[..., sinp.stock['a_nfoet_b1']], -1, b1_pos)
        propn_dst_adj = np.moveaxis(f1_DSTw(repro_rate_adj, cycles = 1)[..., sinp.stock['a_nfoet_b1']], -1, b1_pos)
        ####calculate the change in the expected proportions due to altering the scanning percentage,
        #### only apply to the conception slices - don't want to add any conception to the lambed and lost slices.
        propn_dst_change = (propn_dst_adj - propn_dst) * (nfoet_b1any==nyatf_b1any)
        ####apply the change to the original calculated proportions
        t_cr += propn_dst_change

        ##Process the Conception REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
        ###Conception is the proportion of dams that are dry and a change in conception is assumed to be converting
        ### a dry ewe into a single bearing ewe. It is calculated by altering the proportion of single bearing ewes b1[2].
        ### The proportion of Drys is then calculated (later) as the animals that didn't get pregnant.
        t_cr[:, :, 2, ...] = f1_rev_update('conception', t_cr[:, :, 2, ...], rev_trait_value)

        ##Process the Litter size REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
        ##The litter size REV is stored as the proportion of the pregnant dams that are single-, twin- & triplet-bearing
        ###Steps: Calculate litter size from t_cr, adjust litter size (if required) then recalculate t_cr from new litter size
        ### Calculating litter size (# of foetuses / dam pregnant) requires a mask for the pregnant dams that is the same shape as t_cr
        mask = nfoet_b1any.squeeze() > 0
        t_cr_masked = np.compress(mask, t_cr, b1_pos)
        ### Calculate the proportion of single-, twin- & triplet-bearing dams
        litter_propn = t_cr_masked / np.sum(t_cr_masked, b1_pos, keepdims=True)
        litter_propn = f1_rev_update('litter_size', litter_propn, rev_trait_value)
        ###calculate t_cr from the REV adjusted litter size. t_cr will only change from the original value if litter_size REV is active
        t_cr[:,:,mask,...] = litter_propn * np.sum(t_cr_masked, b1_pos, keepdims=True)

        ##Dams that implant (i.e. do not return to service) but don't retain to 3rd trimester
        ## are added to 00 slice (b1[1:2]) and removed from the NM slice
        ###Number is based on a proportion (cf[5]) of the ewes that implant (propn_preg) losing their foetuses/embryos
        ###The number can't be more than the number of ewes that are not pregnant in the 3rd trimester (1 - propn_preg).
        propn_pregnant = fun.f_dynamic_slice(crg, b1_pos, 2, 3)
        slc[b1_pos] = slice(1,2)
        t_cr[tuple(slc)] = np.minimum((cf[5, ...] / (1 - cf[5, ...])) * propn_pregnant, 1 - propn_pregnant)

        ##If the period is mating then set conception = temporary probability array
        conception = t_cr * period_is_mating

        ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e1[0])
        #todo could this be simplified with the logic conception_e1b1[0,0] = np.sum(conception_e1b1[:, 1:] (as commented below).
        slc = [slice(None)] * len(conception.shape)
        slc[b1_pos] = slice(0,1)
        conception[tuple(slc)] = -np.sum(fun.f_dynamic_slice(conception, b1_pos,1, None), axis = b1_pos, keepdims=True)
        temporary = (index_e1 == 0) * np.sum(conception, axis=sinp.stock['i_e1_pos'], keepdims=True) #sum across e axis into slice e[0]
        conception = fun.f_update(conception, temporary, (nyatf_b1any == 0)) #Put sum of e1 into slice e1[0] and don't overwrite the slices where nyatf != 0

        # slc = [slice(None)] * len(conception.shape)
        # slc[e1_pos] = slice(0, 1)
        # slc[b1_pos] = slice(0, 1)
        # conception[tuple(slc)] = -np.sum(f_dynamic_slice(conception, b1_pos, 1, None), axis=(e1_pos, b1_pos), keepdims=True)
    return conception


def f_conception_ltw(cf, cu0, relsize_mating, cs_mating, scan_std, doy_p, crg_doy, nfoet_b1any, nyatf_b1any, period_is_mating
                     , index_e1, rev_trait_value):
    ''' LTW system: The general calculation is scanning percentage is defined by a linear function of CS
    The standard value (CS 3) is determined by the genotype and relative size
    The slope varies with day of year
    The proportion of dry, single, twin & triplet is estimated as a function of the scanning percentage using f1_DSTw
    Note: sa is not applied in this function because it is applied to the input scan_std (which is also used to determine the BTRT effect on fleece)
    '''
    if ~np.any(period_is_mating):
        conception = np.zeros_like(relsize_mating)
    else:
        b1_pos = sinp.stock['i_b1_pos']  #because used in many places in the function
        ## relative size and condition score of the dams at mating are the determinants of conception
        ### the dams being mated are those in slices e1[0] and b1[0] (first cycle, not mated)
        relsize_mating_e1b1sliced = fun.f_dynamic_slice(relsize_mating, sinp.stock['i_e1_pos'], 0, 1, b1_pos, 0, 1) #take slice e1[0] & b1[0]
        cs_mating_e1b1sliced = fun.f_dynamic_slice(cs_mating, sinp.stock['i_e1_pos'], 0, 1, b1_pos, 0, 1) #take slice e1[0] & b1[0]
        ##Adjust standard scanning percentage based on relative size (to reduce scanning percentage of younger animals)
        scan_std = scan_std * relsize_mating_e1b1sliced * crg_doy
        ##Slope of the RR vs CS relationship based on time of the year
        slope = np.maximum(cu0[4, ...], cu0[2, ...] + np.sin(2 * np.pi * doy_p / 365) * cu0[3, ...])
        ##Reproduction rate for dams as if mated for the number of cycles in the calibration data.
        repro_rate = scan_std + (cs_mating_e1b1sliced - 3) * slope

        ##Calculate the propn dry/single/twin for given repro rate.
        ###remove singleton b1 axis by squeezing because it is replaced by the l0 axis in f1_DSTw)
        repro_rate = np.squeeze(repro_rate, axis=b1_pos)
        ### Note: repro rate calculated above is based on a calibration with 2 cycles
        ### Require the proportions of dry, singles, twins & triplets for 1 cycle
        ### The proportions returned are in axis -1 and needs the slices altered (shape of l0 to b1) and moving to b1 position.
        t_cr = np.moveaxis(f1_DSTw(repro_rate, cycles = 1)[...,sinp.stock['a_nfoet_b1']], -1, b1_pos)
        ##Set proportions to 0 for dams that gave birth and lost - this is required so that numbers in pp behave correctly
        t_cr *= (nfoet_b1any == nyatf_b1any)
        ##Dams that implant (i.e. do not return to service) but don't retain to 3rd trimester are added to 00 slice rather than staying in NM slice
        ###Number is based on a proportion (cf[5]) of the ewes that implant (propn_preg) losing their foetuses/embryos
        ###The number can't be more than the number of ewes that are not pregnant in the 3rd trimester (1 - propn_preg).
        propn_pregnant = np.sum(fun.f_dynamic_slice(t_cr, b1_pos, 2, None), axis = b1_pos, keepdims = True)
        slc = [slice(None)] * len(t_cr.shape)
        slc[b1_pos] = slice(1,2)
        t_cr[tuple(slc)] = np.minimum((cf[5, ...] / (1 - cf[5, ...])) * propn_pregnant, 1 - propn_pregnant)
        ##If the period is mating then set conception = temporary probability array
        conception = t_cr * period_is_mating
        ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e1[0])
        slc = [slice(None)] * len(conception.shape)
        slc[b1_pos] = slice(0,1)
        conception[tuple(slc)] = -np.sum(fun.f_dynamic_slice(conception, b1_pos,1, None), axis = b1_pos, keepdims=True)
        temporary = (index_e1 == 0) * np.sum(conception, axis=sinp.stock['i_e1_pos'], keepdims=True)  # sum across e axis into slice e[0]
        conception = fun.f_update(conception, temporary, (nyatf_b1any == 0))  #Put sum of e1 into slice e1[0] and don't overwrite the slices where nyatf != 0
        ##Process the Conception & Litter size REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
        conception[:, :, 1, ...] = f1_rev_update('conception', conception[:, :, 1, ...], rev_trait_value)
        conception[:, :, 2:, ...] = f1_rev_update('litter_size', conception[:, :, 2:, ...], rev_trait_value)
    return conception


def f1_convert_scancycles(dst_propn, nfoet_b1any, cycles = 1):
    ##Convert from a proportion of dry, singles, twins and triplets from specified number of cycles (usually 1)
    ## to an equivalent scanning percentage if mated for the calibration number of cycles (usually 2)
    ## it is assumed that the litter size is constant and the factor that changes is the proportion of dams that are dry

    #The data used to calibrate the coefficients used are assumed to have been derived from mating for 2 cycles.
    calibration_cycles = 2

    ##scanning percentage for the specified number of cycles
    repro_rate = np.sum(dst_propn * nfoet_b1any, axis = sinp.stock['i_b1_pos'], keepdims = True)

    ##convert number of cycles by scaling the proportion of drys (holding litter size constant)
    dry_propn = fun.f_dynamic_slice(dst_propn, sinp.stock['i_b1_pos'], 1, 2)
    dry_propn_cal = dry_propn ** (calibration_cycles / cycles)
    repro_rate_cal = repro_rate * fun.f_divide(1 - dry_propn_cal, 1 - dry_propn)

    return repro_rate_cal


def f_sire_req(sire_propn_a1e1b1nwzida0e0b0xyg1g0, sire_periods_g0p8, i_sire_recovery, i_startyear, date_end_p, period_is_prejoin_a1e1b1nwzida0e0b0xyg1):
    ##Date at end of period adjusted to start year
    t_date_end_a1e1b1nwzida0e0b0xyg = date_end_p - (365 * (date_end_p.astype('datetime64[Y]').astype(int) + 1970 - i_startyear)).astype('timedelta64[D]')
    ##Date_end falls within the ram mating periods
    sire_required_a1e1b1nwzida0e0b0xyg1g0p8 = np.logical_and(t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] >= sire_periods_g0p8.astype('datetime64[D]') , t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] <= (sire_periods_g0p8.astype('datetime64[D]') + i_sire_recovery)) #add axis for p8 and g1
    ##Number of rams required per ewe (if this period is joining)
    n_sires = sire_required_a1e1b1nwzida0e0b0xyg1g0p8 * sire_propn_a1e1b1nwzida0e0b0xyg1g0[..., na] * period_is_prejoin_a1e1b1nwzida0e0b0xyg1[..., na,na] #add axis for g1 and p8
    return n_sires


##################
#Mortality CSIRO #
##################
'''The CSIRO system includes 
        1.  a base mortality for all animal classes which is a non reducible amount plus an increment
            The increment is a fixed value and occurs if the animals is below a threshold RC and the rate of LWC is below 20% of the normal weight gain
        2. a weaner mortality increment if the animal is less than 365 days old and the rate of LWC is less than 20% of normal weight gain
        3. progeny mortality that is the sum of
            a. mortality due to exposure at birth (mortalityx) that is a function of ewe RC at birth and the chill index at birth
            b. mortality due to difficult birth (mortalityd - dystocia) that depends on the lamb birth weight and ewe relative condition at birth
        4. dam mortality that is the sum of
            a. mortality due to preg toxemia in the last 6 weeks of pregnancy. This occurs for multiple bearing dams and is affected by rate of LW loss
            b. mortality due to dystocia (calculated in f_mortality_progeny_cs). It is assumed that ewe death is associated with a fixed proportion of the lambs deaths from dystocia.
            '''
def f_mortality_base_cs(cd, cg, rc_start, cv_weight, ebg_start, sd_ebg, d_nw_max, days_period, rev_trait_value, sap_mortalityb):
    ## a minimum level of mortality per day that is increased if RC is below a threshold and LWG is below a threshold
    ### i.e. increased mortality only for thin animals that are growing slowly (< 20% of normal growth rate)
    ###distribution on ebg & rc_start, calculate mort and then average (axis =-1,-2)
    ebg_start_p1p2 = fun.f_distribution7(ebg_start, sd=sd_ebg)[...,na]
    rc_start_p1p2 = fun.f_distribution7(rc_start, cv=cv_weight)[...,na,:]
    mortalityb_p1p2 = (cd[1, ...,na,na] + cd[2, ...,na,na] *
                     np.maximum(0, cd[3, ...,na,na] - rc_start_p1p2) *
                     ((cd[16, ...,na,na] * d_nw_max[...,na,na]) > (ebg_start_p1p2 * cg[18, ...,na,na]))) * days_period[...,na,na] #mul by days period to convert from mort per day to per period
    ###average p1 axis
    mortalityb = np.mean(mortalityb_p1p2, axis=(-1,-2))
    ##apply sensitivity
    mortalityb = fun.f_sa(mortalityb, sap_mortalityb, sa_type = 1, value_min = 0)
    ##Process the Mortality REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    mortalityb = f1_rev_update('mortality', mortalityb, rev_trait_value)
    return mortalityb


def f_mortality_weaner_cs(cd, cg, age, ebg_start, sd_ebg, d_nw_max,days_period):
    ## mortality increases (cd[13]) for slow growing young animals (< 20% of normal growth rate).
    ### mortality does not increase with severity of under-nutrition, simply a switch based on growth rate
    ### the mortality increment varies with age. Full increment below 300 days (cd[14]) and ramping down to 0 at 365 days (cd[15])
    ###distribution on ebg - add distribution to ebg_start_p1 and then average (axis =-1)
    ebg_start_p1 = fun.f_distribution7(ebg_start, sd=sd_ebg)
    mort_weaner_p1 = cd[13, ...,na] * fun.f_ramp(age[...,na], cd[15, ...,na], cd[14, ...,na]
                                             ) * ((cd[16, ...,na] * d_nw_max[...,na]
                                                   ) > (ebg_start_p1 * cg[18, ...,na]))* days_period[...,na] #mul by days period to convert from mort per day to per period
    return np.mean(mort_weaner_p1, axis=-1)

def f_mortality_dam_cs():
    '''
    Peri natal (at birth) Dam mortality.
    Currently the CSIRO system includes dam mortality due to PregTox (f_mortality_pregtox_cs) and dam mortality due to
    dystocia (included in f_mortality_progeny_cs) but there is no calculation of deaths from other causes,
    such as those that might effect twin bearing dams at birth.
    '''
    return 0


def f_mortality_pregtox_cs(cb1, cg, nw_start, ebg, sd_ebg, days_period, period_between_birth6wks, gest_propn, sap_mortalitye):
    '''
    (Twin) Dam mortality in last 6 weeks (preg tox). This increments mortality associated with LWL in the base mortality function.

    Preg tox is short for pregnancy toxaemia. It is associated with ketosis where the ewe switches into burning fat
    (rather than carbohydrates) because they are losing weight. It is predominantly a problem for twin bearing ewes
    because they have the highest energy demands close to lambing and the least capacity to eat more (because their
    insides are full of lambs and this restricts stomach capacity). It is usually also more of a problem for ewes
    that start out in better condition.
    '''
    ###distribution on ebg - add distribution to ebg_start_p1 and then average (axis =-1)
    ebg_p1 = fun.f_distribution7(ebg, sd=sd_ebg)
    t_mort_p1 = days_period[..., na] * gest_propn[..., na] / 42 * fun.f_sig(-42 * ebg_p1 * cg[18, ..., na] / nw_start[..., na]
                                                                        , cb1[4, ..., na], cb1[5, ..., na]) #mul by days period to convert from mort per day to per period
    t_mort = np.mean(t_mort_p1, axis=-1)
    ##If not last 6 weeks then = 0
    mort = t_mort * period_between_birth6wks
    ##Adjust by sensitivity on dam mortality
    mort = fun.f_sa(mort, sap_mortalitye, sa_type = 1, value_min = 0)
    return mort

    
def f_mortality_progeny_cs(cd, cb1, w_b, rc_birth, cv_weight, w_b_exp_y, period_is_birth, chill_index_p1, nfoet_b1
                           , rev_trait_value, sap_mortalityp, saa_mortalityx):
    '''Progeny losses due to large progeny or slow birth process (dystocia)

    Dystocia definition is a difficult birth that leads to brain damage, which can be due to physical trauma
    but also lack of oxygen. The difficult birth can be a larger single lamb but it is also quite prevalent
    in twins not due to large lambs but due to a slow birth because the ewe is lacking energy to push.
    '''
    ###distribution on w_b & rc_birth - add distribution to ebg_start_p1 and then average (axis =-1)
    w_b_p1p2 = fun.f_distribution7(w_b, cv=cv_weight)[...,na]
    rc_birth_p1p2 = fun.f_distribution7(rc_birth, cv=cv_weight)[...,na,:]
    mortalityd_yatf_p1p2 = fun.f_sig(fun.f_divide(w_b_p1p2, w_b_exp_y[...,na,na]) * np.maximum(1, rc_birth_p1p2),
                                 cb1[6, ...,na,na], cb1[7, ...,na,na]) * period_is_birth[...,na,na]
    mortalityd_yatf = np.mean(mortalityd_yatf_p1p2, axis=(-1,-2))
    ##add sensitivity
    mortalityd_yatf = fun.f_sa(mortalityd_yatf, sap_mortalityp, sa_type = 1, value_min = 0)
    ##dam mort due to large progeny or lack of energy at birth (dystocia) - returns 0 mort if there is 0 nfoet also the fact that more prog die per dam when the dams has multiple nfoet (eg for a trip only one ewe dies for every 3 yatf)
    mortalityd_dams = fun.f_divide(np.mean(mortalityd_yatf, axis=sinp.stock['i_x_pos'], keepdims=True) * cd[21,...], nfoet_b1)
    ##Reduce progeny losses due to large progeny (dystocia) - so not double counting progeny losses associated with dam mortality
    mortalityd_yatf = mortalityd_yatf * (1- cd[21,...])
    ##Exposure index
    xo_p1p2 = cd[8, ..., na,na] - cd[9, ..., na,na] * rc_birth_p1p2 + cd[10, ..., na,na] * chill_index_p1[..., na] + cb1[11, ..., na,na]
    ##Progeny mortality at birth from exposure
    mortalityx = np.average(np.exp(xo_p1p2) / (1 + np.exp(xo_p1p2)) ,axis = (-1,-2)) * period_is_birth #axis -1 is p1
    ##Apply SA to progeny mortality due to exposure
    mortalityx = fun.f_sa(mortalityx, sap_mortalityp, sa_type = 1, value_min = 0)
    mortalityx = fun.f_sa(mortalityx, saa_mortalityx, sa_type = 2, value_min = 0)
    ##Process the Ewe Rearing Ability REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    mortalityx = f1_rev_update('era', mortalityx, rev_trait_value)
    return mortalityx, mortalityd_yatf, mortalityd_dams

####################
#Mortality Murdoch #
####################
''' 
    The Murdoch Uni system includes:
        1. a base mortality for all animal classes which is a non reducible amount plus an increment
           The increment varies quadratically with both RC if below a threshold and ebg if below a threshold (relative to normal weight change)
        2. Weaner mortality is included in the base mortality through ebg being compared with normal growth rate.
        3. progeny mortality calculated from the LTW equations and is a function of birth weight, birth type and chill index at birth
        4. dam mortality: 
            a. f_mortality_dam_mu - a function of dam CS at birth. This is the increase in mortality for reproducing ewes and 
               is the same for single and twin bearing ewes.
            b. f_mortality_pregtox_cs - currently this is just 0 it needs to be built.
'''
def f_mortality_base_mu(cd, cg, rc_start, cv_weight, ebg_start, sd_ebg, d_nw_max, days_period, rev_trait_value, sap_mortalityb):
    ## a minimum level of mortality per day that is increased if RC is below a threshold and LWG is below a threshold
    ### the mortality rate increases in a quadratic function for lower RC & greater disparity between EBG and normal gain
    ###distribution on ebg & rc_start, calculate mort and then average (axis =-1,-2)
    ebg_start_p1p2 = fun.f_distribution7(ebg_start, sd=sd_ebg)[...,na]
    rc_start_p1p2 = fun.f_distribution7(rc_start, cv=cv_weight)[...,na,:]
    ###calc mort scalars
    rc_mortality_scalar_p1p2 = (np.minimum(0, rc_start_p1p2 - cd[24, ...,na,na])
                                / (cd[23, ...,na,na] - cd[24, ...,na,na]))**2
    ebg_mortality_scalar_p1p2 = (np.minimum(0, ebg_start_p1p2 * cg[18, ...,na,na] - cd[26, ...,na,na] - d_nw_max[...,na,na])
                                 / (cd[25, ...,na,na] - cd[26, ...,na,na]))**2
    mortalityb_p1p2 = (cd[1, ...,na,na] + cd[22, ...,na,na] * rc_mortality_scalar_p1p2 * ebg_mortality_scalar_p1p2) * days_period[...,na,na]  #mul by days period to convert from mort per day to per period
    mortalityb = np.mean(mortalityb_p1p2, axis=(-1,-2))
    ##apply sensitivity
    mortalityb = fun.f_sa(mortalityb, sap_mortalityb, sa_type = 1, value_min = 0)
    ##Process the Mortality REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    mortalityb = f1_rev_update('mortality', mortalityb, rev_trait_value)
    return mortalityb


def f_mortality_weaner_mu():
    ## The MU base mortality function accounts for the mortality increases for slow growing young animals
    #todo incorporate Angus Campbell's mortality function as the MU weaner mortality function (to replace the base mortality for weaners)
    return 0


def f_mortality_dam_mu(cu2, cs_birth_dams, cv_cs, period_is_birth, nfoet_b1, sap_mortalitye):
    ## transformed Dam mortality at birth due to low CS.
    ###distribution on cs_birth, calculate mort and then average (axis =-1)
    cs_birth_dams_p1 = fun.f_distribution7(cs_birth_dams, cv=cv_cs)
    ###calc mort
    t_mortalitye_mu_p1 = cu2[22, 0, ...,na] * cs_birth_dams_p1 + cu2[22, 1, ...,na] * cs_birth_dams_p1 ** 2 + cu2[22, -1, ...,na]
    ##Back transform the mortality
    mortalitye_mu_p1 = np.exp(t_mortalitye_mu_p1) / (1 + np.exp(t_mortalitye_mu_p1)) * period_is_birth[...,na]
    mortalitye_mu = np.mean(mortalitye_mu_p1, axis=-1)
    ##no increase in mortality for the non reproducing ewes (n_foet == 0)
    mortalitye_mu = mortalitye_mu * (nfoet_b1 > 0)
    ##Adjust by sensitivity on dam mortality
    mortalitye_mu = fun.f_sa(mortalitye_mu, sap_mortalitye, sa_type = 1, value_min = 0)
    return mortalitye_mu


def f_mortality_pregtox_mu():
    '''
    (Twin) Dam mortality in last 6 weeks (preg tox).

    Preg tox is short for pregnancy toxaemia. It is associated with ketosis where the ewe switches into burning fat
    (rather than carbohydrates) because they are losing weight. It is predominantly a problem for twin bearing ewes
    because they have the highest energy demands close to lambing and the least capacity to eat more (because their
    insides are full of lambs and this restricts stomach capacity). It is usually also more of a problem for ewes
    that start out in better condition.
    '''
    #todo hook this up with relationships developed in Lifetime maternals project
    return 0

def f_mortality_progeny_mu(cu2, cb1, cx, ce, w_b, w_b_std, cv_weight, foo, chill_index_p1, period_is_birth
                           , rev_trait_value, sap_mortalityp, saa_mortalityx):
    '''
    Calculate the mortality of progeny at birth due to mis-mothering and exposure
    using the LTW prediction equations (Oldham et al. 2011) with inclusion of chill index.
    The paddock level scalar is added (Young et al 2011) however, this is not calibrated for high chill environments (>1000)
    The scalar adjusts the difference in survival if birth weight is different from the standard birthweight
    this is to reflect the difference in survival observed in the LTW paddock trial compared with the plot scale trials.
    '''
    ##transformed survival for actual & standard
    ###distribution on w_b & rc_birth - add distribution to ebg_start_p1 and then average (axis =-1)
    w_b_p1p2 = fun.f_distribution7(w_b, cv=cv_weight)[...,na,:]
    w_b_std_p1p2 = fun.f_distribution7(w_b_std, cv=cv_weight)[...,na,:]

    t_survival_p1p2 = (cu2[8, 0, ...,na,na] * w_b_p1p2 + cu2[8, 1, ..., na,na] * w_b_p1p2 ** 2
                      + cu2[8, 2, ..., na,na] * chill_index_p1[...,na] + cu2[8, 3, ..., na,na] * foo[..., na,na]
                      + cu2[8, 4, ..., na,na] * foo[..., na,na] ** 2 + cu2[8, 5, ..., na,na] + cb1[8, ..., na,na]
                      + cx[8, ..., na,na] + cx[9, ..., na,na] * chill_index_p1[...,na] + ce[8, ..., na,na])
    t_survival_std_p1p2 = (cu2[8, 0, ..., na,na] * w_b_std_p1p2 + cu2[8, 1, ..., na,na] * w_b_std_p1p2 ** 2
                      + cu2[8, 2, ..., na,na] * chill_index_p1[...,na] + cu2[8, 3, ..., na,na] * foo[..., na,na]
                      + cu2[8, 4, ..., na,na] * foo[..., na,na] ** 2 + cu2[8, 5, ..., na,na] + cb1[8, ..., na,na]
                      + cx[8, ..., na,na] + cx[9, ..., na,na] * chill_index_p1[...,na] + ce[8, ..., na,na])
    ##back transformed & converted to mortality
    mortalityx = (1 - np.average(1 / (1 + np.exp(-t_survival_p1p2)),axis = (-1,-2))) * period_is_birth #p1 axis averaged
    mortalityx_std = (1 - np.average(1 / (1 + np.exp(-t_survival_std_p1p2)),axis = (-1,-2))) * period_is_birth #p1 axis averaged
    ##Scale progeny survival using paddock level scalars
    mortalityx = mortalityx_std + (mortalityx - mortalityx_std) * cb1[9, ...]
    ##Apply SA to progeny mortality at birth (LTW)
    mortalityx = fun.f_sa(mortalityx, sap_mortalityp, sa_type = 1, value_min = 0)
    mortalityx = fun.f_sa(mortalityx, saa_mortalityx, sa_type = 2, value_min = 0)
    ##Process the Ewe Rearing Ability REV: either save the trait value to the dictionary or over write trait value with value from the dictionary
    mortalityx = f1_rev_update('era', mortalityx, rev_trait_value)
    return mortalityx

#######################
#end of loop functions#
#######################

def f1_period_start_prod(numbers, var, prejoin_tup, season_tup, period_is_startseason, mask_min_lw_z, period_is_prejoin=0, group=None):
    ##Set variable level = value at end of previous	
    var_start = var
    ##make sure numbers and var are same shape - this is required for the np.average func below
    numbers, var_start = np.broadcast_arrays(numbers,var_start)

    ##a) Calculate temporary values as if period is start of season
    if np.any(period_is_startseason):
        var_start = f1_season_wa(numbers, var_start, season_tup, mask_min_lw_z, period_is_startseason)

    ##b) Calculate temporary values as if period_is_prejoin
    if group==1 and np.any(period_is_prejoin):
        temporary = fun.f_weighted_average(var_start, numbers, prejoin_tup, keepdims=True, non_zero=True) #gets the weighted average of production in the different seasons
        ##Set values where it is beginning of FVP
        var_start = fun.f_update(var_start, temporary, period_is_prejoin)
    return var_start


def f1_season_wa(numbers, var, season, mask_min_lw_z, period_is_startseason):
    '''
    Perform weighted average across seasons, at the beginning of each season.
    So all seasons start from a common place.
    The animals with the lightest liveweight patterns (there could be multiple because depending on the fvp the w axis may be clustered)
     at the time of season start are assigned the lowest live weight from across the z axis rather than the weighted average,
     so that light animals are not lost in the postprocessing distribution.
    Don't need to worry about mortality in the different slices because this is not to do with condensing (in condensing we take the weights of animals with less than 10% mort).
    '''
    temporary = fun.f_weighted_average(var,numbers,season,keepdims=True, non_zero=True)  # gets the weighted average of production in the different seasons
    ###adjust production for min lw: the w slices with the minimum lw get assigned the production associated with the animal from the season with the lightest animal (this is so the light animals in the poor seasons are not disregarded when distributing in PP).
    ###use masked array to average the production from the z slices with the lightest animal (this is required in case multiple z slices have the same weight animals)
    masked_var = np.ma.masked_array(var,np.logical_not(mask_min_lw_z))
    mean_var = np.mean(masked_var ,axis=season,keepdims=True) #take the mean in case multiple season slices have the same weight light animal.
    temporary[np.any(mask_min_lw_z,axis=season, keepdims=True)] = mean_var[np.any(mask_min_lw_z,axis=season, keepdims=True)]

    ###Set values where it is beginning of FVP
    var = fun.f_update(var,temporary,period_is_startseason)
    return var


def f1_condensed(var, lw_idx, condense_w_mask, i_n_len, i_w_len, i_n_fvp_period, period_is_condense):
    """
    Condense variable to x common points along the w axis when period_is_condense.
    Currently this function only handle 2 or 3 initial liveweights. The order of the returned W axis is M, H, L for 3 initial lws or H, L for 2 initial lws.

    :param numbers: current end numbers
    :param var: production variable being condensed
    :param lw_idx: index specifying the sorted order of the w axis
    :param prejoin_tup: prejoin axis
    :param season_tup: season axis
    :param i_n_len: number of nutrition options
    :param i_w_len: length of w axis
    :param i_n_fvp_period: number of fvps
    :param numbers_start_condense: numbers at the previous condensing (used to calc mortality)
    :param period_is_condense: bool array
    :return:
    """
    if np.any(period_is_condense):
        var = np.broadcast_to(var,lw_idx.shape).copy() #need to copy so array can be assigned to down below
        temporary = np.zeros_like(var)  #this is done to ensure that temp has the same size as var.
        ##test if array has diagonal and calc temp variables as if start of dvp - if there is not a diagonal use the alternative system for reallocating at the end of a DVP
        if i_n_len >= i_w_len:
            ###this method was the way we first tried - no longer used (might be used later if we add nutrient options back in)
            ### np.diagonal removes the n axis so it is added back in using the expand function, but that is a singleton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
            temporary[...] = np.expand_dims(np.rollaxis(temporary.diagonal(axis1= sinp.stock['i_w_pos'], axis2= sinp.stock['i_n_pos']),-1,sinp.stock['i_w_pos']), sinp.stock['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
        else:
            '''
            possible idea to handle more than 3 starting lws.
            Note: it is good to keep the medium lw as slice 0 because that means it is held the same across all dvps 
             and thus the user can make its pattern optimal.
                        
            if n_initial_lws==2:
                #in code already. Nothing needs to be changed for this part                
            
            else: #need to update the code with something like this.
                for i in n_initial_lws:
                #use i to assign and select correct patterns
                    if i == 0 or >2:
                        #medium = use current code with i to select correct pattern and assign
                        sl = [slice(None)] * temporary.ndim
                        sl_start_med = i*int(i_n_len ** i_n_fvp_period)
                        sl_end_med = (i+1)*int(i_n_len ** i_n_fvp_period)
                        sl[sinp.stock['i_w_pos']] = slice(sl_start_med, sl_end_med)
                        if i_n_len >= 3:
                            temporary[tuple(sl)] = f_dynamic_slice(var, sinp.stock['i_w_pos'], sl_start_med, sl_start_med+1)  # the pattern that is feed supply 1 (median) for the entire year (the top w pattern)
                        else:
                            temporary[tuple(sl)] = np.mean(var_sorted_mort, axis=sinp.stock['i_w_pos'], keepdims=True) ^this line wont work if more than 3 lws, somehow need to take mean of subsection of w axis depending on number of initial w. # average of all animals with less than 10% mort

                    elif i==1:
                        #calc high - using sorted var
                        sl = [slice(None)] * temporary.ndim
                        sl[sinp.stock['i_w_pos']] = slice(i*int(i_n_len ** i_n_fvp_period), (i+1)*int(i_n_len ** i_n_fvp_period))
                        temporary[tuple(sl)] = np.mean(f_dynamic_slice(var_sorted, sinp.stock['i_w_pos'], i_w_len - int(i_w_len / 10), -1), sinp.stock['i_w_pos'], keepdims=True)  # average of the top lw patterns

                    else: #i==2 (low w)
                        #calc low
                        numbers_start_sorted = np.take_along_axis(numbers_start_condense, lw_idx, axis=sinp.stock['i_w_pos'])
                        numbers_sorted = np.take_along_axis(numbers, lw_idx, axis=sinp.stock['i_w_pos'])
                        low_slice = np.argmax(np.sum(numbers_start_sorted, axis=prejoin_tup + (season_tup,), keepdims=True)
                                                     / np.sum(numbers_sorted, axis=prejoin_tup + (season_tup,), keepdims=True) > 0.9
                                                     , axis=sinp.stock['i_w_pos'])  # returns the index of the first w slice that has mort less the 10%.
                        low_slice = np.expand_dims(low_slice, axis=sinp.stock['i_w_pos']) #add singleton w axis back
                        sl = [slice(None)] * temporary.ndim
                        sl[sinp.stock['i_w_pos']] = slice(i*int(i_n_len ** i_n_fvp_period), (i+1)*int(i_n_len ** i_n_fvp_period))
                        temporary[tuple(sl)] = np.take_along_axis(var_sorted, low_slice, sinp.stock['i_w_pos'])
                                     
            '''

            ###sort var based on animal lw
            ma_var = np.ma.masked_array(var, np.logical_not(condense_w_mask))
            ma_var_sorted = np.take_along_axis(ma_var, lw_idx, axis=sinp.stock['i_w_pos']) #sort into production order (base on lw) so we can select the production of the lowest lw animals with mort less than 10% - note sorts in ascending order

            ###count number of w slices which have been masked out
            idx_min_lw = np.count_nonzero(condense_w_mask==0, axis=sinp.stock['i_w_pos'], keepdims=True) #index of the min lw with mort less than 10%
            n_lw = np.min(np.count_nonzero(condense_w_mask, axis=sinp.stock['i_w_pos'])) #number of included w slices

            ##to handle varying number of initial lws
            if sinp.structuralsa['i_w_start_len1'] == 2:
                ###add high pattern - this will not handle situations where the top 10% lw animals all have higher mortality than the threshold.
                temporary[...] = np.mean(
                    fun.f_dynamic_slice(ma_var_sorted,sinp.stock['i_w_pos'],i_w_len - int(math.ceil(n_lw / 10)), None), # ceil is used to handle cases where nutrition options is 1 (eg only 3 lw patterns)
                    sinp.stock['i_w_pos'],keepdims=True)  # average of the top lw patterns

                ###low pattern - production level of the lowest nutrition profile that has a mortality less than 10% for the year
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(-int(i_n_len ** i_n_fvp_period), None)
                temporary[tuple(sl)] = np.take_along_axis(ma_var_sorted, idx_min_lw, sinp.stock['i_w_pos']) #if you get an error here it probably means no animals had mort less than 10%

            else:
                ###add high pattern - this will not handle situations where the top 10% lw animals all have higher mortality than the threshold.
                temporary[...] = np.mean(fun.f_dynamic_slice(ma_var_sorted, sinp.stock['i_w_pos'], i_w_len - int(math.ceil(n_lw / 10)), None),  #ceil is used to handle cases where nutrition options is 1 (eg only 3 lw patterns)
                                         sinp.stock['i_w_pos'], keepdims=True)  # average of the top lw patterns

                ###add mid pattern (w 0 - 27) - use slice method in case w axis changes position (can't use MRYs dynamic slice function because we are assigning)
                ###if there is 3n then medium condense is the top slice (medium start weight with medium nutrition). It is best to keep the middle w to slice 0 rather than the average because then medium always passes to medium so the user can attempt to more easily optimise the nutrition for medium lw.
                ###if there is 2n then medium condense is the average of all animals with less than 10% mort.
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(0, int(i_n_len ** i_n_fvp_period))
                if i_n_len == 2:
                    temporary[tuple(sl)] = np.mean(ma_var_sorted, axis=sinp.stock['i_w_pos'], keepdims=True)  # average of all animals with less than 10% mort
                else:
                    temporary[tuple(sl)] = fun.f_dynamic_slice(var, sinp.stock['i_w_pos'], 0, 1)  # the pattern that is feed supply 1 (median) for the entire year (the top w pattern)

                ###low pattern - production level of the lowest nutrition profile that has a mortality less than 10% for the year
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(-int(i_n_len ** i_n_fvp_period), None)
                temporary[tuple(sl)] = np.take_along_axis(ma_var_sorted, idx_min_lw, sinp.stock['i_w_pos']) #if you get an error here it probably means no animals had mort less than 10%
        ###Update if the period is start of year (shearing for offs and prejoining for dams)
        var = fun.f_update(var, temporary, period_is_condense)

    return var


def f1_period_start_nums(numbers, prejoin_tup, season_tup, period_is_startseason, season_propn_z, group=None, nyatf_b1 = 0
                        , numbers_initial_repro=0, gender_propn_x=1, period_is_prejoin=0, period_is_birth=False):
    ##a) reallocate for season type
    if np.any(period_is_startseason):
        temporary = np.sum(numbers, axis = season_tup, keepdims=True)  * season_propn_z  #Calculate temporary values as if period_is_break
        numbers = fun.f_update(numbers, temporary, period_is_startseason)  #Set values where it is beginning of FVP
    ##b)things for dams - prejoining and moving between classes
    if group==1 and np.any(period_is_prejoin):
        ##d) new repro cycle (prejoining)
        temporary = np.sum(numbers, axis = prejoin_tup, keepdims=True) * numbers_initial_repro #Calculate temporary values as if period_is_prejoin
        numbers = fun.f_update(numbers, temporary, period_is_prejoin)  #Set values where it is beginning of FVP
    ##c)things just for yatf
    if group==2:
        temp = nyatf_b1 * gender_propn_x   # nyatf is accounting for peri-natal mortality. But doesn't include the differential mortality of female and male offspring at birth
        numbers=fun.f_update(numbers, temp, period_is_birth)
    return numbers


def f1_period_end_nums(numbers, mortality, numbers_min_b1, mortality_yatf=0, nfoet_b1 = 0, nyatf_b1 = 0, group=None
                      , conception = 0, scan=0, gbal=0, gender_propn_x=1, period_is_mating = False
                      , period_is_matingend = False, period_is_birth=False, period_is_scan=False, propn_dams_mated=1):
    '''
    This adjusts numbers for things like conception and mortality that happen during a given period
    '''
    ##a) mortality (include np.maximum on mortality so that numbers can't become negative)
    numbers = numbers * np.maximum(0, 1-mortality)
    ##numbers for post processing - don't include selling drys - assignment required here for when it is not group 1 or 2
    pp_numbers = numbers
    ##things for dams - prejoining and moving between classes
    if group==1:
        ###b) conception - conception is the change in numbers +ve for animals getting pregnancy and -ve in the NM e-0 slice (note the conception for e slice 1 and higher puts the negative numbers in the e-0 nm slice)
        if np.any(period_is_mating):
            temporary = numbers + conception * numbers[:, 0:1, 0:1, ...]  # numbers_dams[:,0,0,...] is the NM slice of cycle 0 ie the number of animals yet to be mated (conception will have negative value in nm slice)
            numbers = fun.f_update(numbers, temporary, np.any(period_is_mating, axis=sinp.stock['i_e1_pos'])) #needs to be previous period else conception is not calculated because numbers happens at beginning of p loop
        ###at the end of mating move any remaining numbers from nm to 00 slice (note only the nm slice for e-0 has numbers - this is handled in the conception function)
        ###Set temporary to copy of current numbers
        if np.any(period_is_matingend):
            temporary  = np.copy(numbers)
            temporary[:, 0:1, 1:2, ...] += numbers[:, 0:1, 0:1, ...]   # add the number remaining unmated to the dry slice in e1[0]
            temporary[:, :, 0:1, ...] = 0 #set the NM slice to 0 (because they have just been added to drys)
            ##handle the proportion mated. Note: if the inputs are set to optimise the proportion (np.inf) then it is treated as 100% mated
            mated_propn = np.minimum(1, propn_dams_mated) #maximum value of 1 because default is inf, otherwise propn to be mated.
            ### the number in the NM slice e1[0] is a proportion of the total numbers
            ### need a minimum number otherwise get nan later. Want a small number relative to mortality (after allowing for multiple slices getting the small number)
            temporary[:, 0:1, 0:1, ...] = np.maximum(0.00001, np.sum(temporary, axis=(sinp.stock['i_e1_pos'], sinp.stock['i_b1_pos']),
                                                                     keepdims=True) * (1 - mated_propn))
            ### the numbers in the other mated slices other than NM get scaled by the proportion mated
            temporary[:, :, 1:, ...] = np.maximum(0, temporary[:, :, 1:, ...] * mated_propn)
            ###update numbers with the temporary calculations if it is the end of mating
            numbers = fun.f_update(numbers, temporary, period_is_matingend)
        ###d) birth (account for birth status and if drys are retained)
        if np.any(period_is_birth):
            dam_propn_birth_b1 = fun.f_comb(nfoet_b1, nyatf_b1) * (1 - mortality_yatf) ** nyatf_b1 * mortality_yatf ** (nfoet_b1 - nyatf_b1) # the proportion of dams of each LSLN based on (progeny) mortality
            ##have to average x axis so that it is not active for dams - times by gender propn to give approx weighting (ie because offs are not usually entire males so they will get low weighting)
            temp = np.sum(dam_propn_birth_b1 * gender_propn_x, axis=sinp.stock['i_x_pos'], keepdims=True) * numbers[:,:,sinp.stock['ia_prepost_b1'],...]
            pp_numbers = fun.f_update(numbers, temp, period_is_birth)  # calculated in the period after birth when progeny mortality due to exposure is calculated
            temp = np.maximum(pinp.sheep['i_drysretained_birth'],np.minimum(1, nyatf_b1)) * pp_numbers
            numbers = fun.f_update(pp_numbers, temp, period_is_birth * (gbal>=2)) # has to happen after the dams are moved due to progeny mortality so that gbal drys are also scaled by drys_retained
        else:
            ##numbers for post processing - don't include selling drys - assignment required here in case it is not birth
            pp_numbers = numbers
        ###c) scanning
        if np.any(period_is_scan):
            temp = np.maximum(pinp.sheep['i_drysretained_scan'],np.minimum(1, nfoet_b1)) * numbers # scale the level of drys by drys_retained, scale every other slice by 1 except drys if not retained
            numbers = fun.f_update(numbers, temp, period_is_scan * (scan>=1))
        ###e)make the max of number 0.0001
        numbers=np.maximum(numbers_min_b1, numbers)
    return numbers,pp_numbers




#################
#post processing#
#################
def f_wool_additional(fd, sl, ss, vm,  pmb, cvfd=0.22, cvsl=0.18):
    cu5_u5c5=uinp.sheep['i_cu5_c5']
    i_eqn_ph=uinp.sheep['i_eqn_ph']
    i_eqn_cvh=uinp.sheep['i_eqn_cvh']
    i_eqn_romaine=uinp.sheep['i_eqn_romaine']
    ##adjusted pmb
    pmb = np.maximum(cu5_u5c5[4, i_eqn_ph], pmb)
    ##predicted hauteur price adj
    ph = cu5_u5c5[0, i_eqn_ph] * sl + cu5_u5c5[1, i_eqn_ph] * ss + cu5_u5c5[2, i_eqn_ph] * fd + cu5_u5c5[
        3, i_eqn_ph] * pmb + cu5_u5c5[5, i_eqn_ph] * vm + cu5_u5c5[6, i_eqn_ph] * cvfd + cu5_u5c5[
          7, i_eqn_ph] * cvsl + cu5_u5c5[8, i_eqn_ph]
    ##Back transform the ph if using CSIRO equation
    if i_eqn_ph == 0:
        ph = 1 / (1 + np.exp(-ph))
    ##predicted cv hauteur
    cvh = cu5_u5c5[0, i_eqn_cvh] * sl + cu5_u5c5[1, i_eqn_cvh] * ss + cu5_u5c5[2, i_eqn_cvh] * fd + cu5_u5c5[
        3, i_eqn_cvh] * pmb + cu5_u5c5[5, i_eqn_cvh] * vm + cu5_u5c5[6, i_eqn_cvh] * cvfd + cu5_u5c5[
          7, i_eqn_cvh] * cvsl + cu5_u5c5[8, i_eqn_cvh]
    ##predicted romaine
    romaine = cu5_u5c5[0, i_eqn_romaine] * sl + cu5_u5c5[1, i_eqn_romaine] * ss + cu5_u5c5[2, i_eqn_romaine] * fd + \
      cu5_u5c5[3, i_eqn_romaine] * pmb + cu5_u5c5[5, i_eqn_romaine] * vm + cu5_u5c5[6, i_eqn_romaine] * cvfd + \
      cu5_u5c5[7, i_eqn_romaine] * cvsl + cu5_u5c5[8, i_eqn_romaine]
    return ph, cvh, romaine


def f1_woolprice():
    '''Calculate the micron price guide (MPG) for a range of FD (where the FD is of the fleece component of the clip)
    Includes sensitivity on the average micron price guide (MPG) and the premium for finer wool
    The inputs for the function are:
    i_woolp_mpg_range_w5 - the percentile values for which the MPG is input
    i_woolp_mpg_w5 - the MPG for the base FD at each of the percentile levels
    i_woolp_mpg_percentile - the percentile level to use for this trial
    i_woolp_fdprem_range_w5 - the percentile values for which the FD premium is input
    i_woolp_fdprem_w4w5 - the FD premium at each of the percentile levels
    i_woolp_fdprem_percentile - the percentile level to use for this trial
    '''
    ##input value for micron price guide percentile to use (adjusted by SAV during the input process)
    mpg_percentile = uinp.sheep['i_woolp_mpg_percentile']
    ##extrapolate price for the std FD at selected percentile (can go beyond the data input range)
    mpg_stdfd = fun.np_extrap(mpg_percentile, uinp.sheep['i_woolp_mpg_range_w5'], uinp.sheep['i_woolp_mpg_w5'])
    ##adjust price for the std FD using sav
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sav['woolp_mpg'], 5)
    ##adjust price for the std FD using sam
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sam['woolp_mpg'])
    ##FD percentile to use (adjusted by SAV during the input process)
    fd_percentile = uinp.sheep['i_woolp_fdprem_percentile']
    ##extrapolate FD premium at selected percentile for each FD (can go beyond the data input range)
    fdprem_w4 = np.array([fun.np_extrap(fd_percentile, uinp.sheep['i_woolp_fdprem_range_w5'], uinp.sheep['i_woolp_fdprem_w4w5'][i])
                          for i in range(uinp.sheep['i_woolp_fdprem_w4w5'].shape[0])])
    ##adjust FD premium using sav
    fdprem_w4 = fun.f_sa(fdprem_w4, sen.sav['woolp_fdprem'], 5)
    ##Wool price for the analysis (Note: fdprem is the premium per micron from the base)
    mpg_w4 = mpg_stdfd * (1 + fdprem_w4) ** (uinp.sheep['i_woolp_fd_std'] - uinp.sheep['i_woolp_fd_range_w4'])
    return mpg_w4


def f_wool_value(mpg_w4, cfw_pg, fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg,dtype=None):
    '''Calculate the net value of the wool on the sheep's back (cost of shearing is not included in these calculations)
    Includes adjusting price for FD, level of fault (VM & predicted hauteur) and all components of the clip (STB)
    FNF is 'free or nearly free' i.e. wool with no fault (low VM & high SS)
    STB is sweep the board i.e. including all the wool types that are produced (fleece, pieces, bellies ...)
    NIB is net in the bank i.e. all selling, testing & freight costs removed
    '''
    ##call function to calculate predicted hauteur (ph), CV of hauteur (cvh) and romaine
    ph_pg, cvh_pg, romaine_pg = f_wool_additional(fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg)
    ##STB price for FNF (free or nearly free of fault)
    fnfstb_pg = np.interp(fd_pg, uinp.sheep['i_woolp_fd_range_w4'], mpg_w4 * uinp.sheep['i_stb_scalar_w4']).astype(dtype)
    ##vm price adj
    vm_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_vm_adj_w4w6'], uinp.sheep['i_woolp_vm_range_w6']
                                           , uinp.sheep['i_woolp_fd_range_w4'], vm_pg,fd_pg).astype(dtype)
    ##predicted hauteur price adj
    ph_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_ph_adj_w4w7'], uinp.sheep['i_woolp_ph_range_w7']
                                           , uinp.sheep['i_woolp_fd_range_w4'], ph_pg,fd_pg).astype(dtype)
    ##cv hauteur price adj
    cvh_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_cvh_adj_w4w8'], uinp.sheep['i_woolp_cvh_range_w8']
                                            , uinp.sheep['i_woolp_fd_range_w4'], cvh_pg,fd_pg).astype(dtype)
    ##romaine price adj
    romaine_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_romaine_adj_w4w9'], uinp.sheep['i_woolp_romaine_range_w9']
                                                , uinp.sheep['i_woolp_fd_range_w4'], romaine_pg,fd_pg).astype(dtype)
    ##wool price with adjustments
    woolp_stb_pg = fnfstb_pg * (1 + vm_adj_pg) * (1 + ph_adj_pg) * (1 - cvh_adj_pg) * (1 - romaine_adj_pg)
    ##stb net in the bank price
    woolp_stbnib_pg = woolp_stb_pg * (1 - uinp.sheep['i_wool_cost_pc']) - uinp.sheep['i_wool_cost_kg']
    ##wool value if shorn this period
    wool_value_pg = woolp_stbnib_pg * cfw_pg
    return wool_value_pg, woolp_stbnib_pg

def f1_condition_score(rc, cu0):
    ''' Estimate CS from LW. Works with scalars or arrays - provided they are broadcastable into ffcfw.

       ffcfw: (kg) Fleece free, conceptus free liveweight. normal_weight: (kg). cs_propn: (0.19) change in LW
       associated with 1 CS as a proportion of normal_weight.

       long version of the formula (use rc instead of using to following): 3 + (ffcfw - normal_weight) / (cs_propn * normal_weight)
       Returns: condition score - float
       '''
    return np.maximum(1, 3 + (rc - 1) / cu0[1, ...]) #a minimum value of CS=1 is used to remove errors caused by low CS. A CS below 1 is unlikely because the animal would be dead


#todo needs updating - currently just a copy of the cs function
def f1_fat_score(rc, cu0):
    return np.maximum(1, 3 + (rc - 1) / cu0[1, ...]) #FS 1 is the lowest possible measurement. FS1 is between 0 and 5mm of tissue at the GR site.


def f1_saleprice(score_pricescalar_s7s5s6, weight_pricescalar_s7s5s6, dtype=None):
    ##Sale price percentile to use (adjusted by sav)
    salep_percentile = uinp.sheep['i_salep_percentile']
    ##Max price in each grid (s7 axis) at selected percentile - 1d extrapolation along the s4 axis (can go beyond the input range)
    grid_max_s7 = (np.array([fun.np_extrap(salep_percentile, uinp.sheep['i_salep_percentile_range_s4'], uinp.sheep['i_salep_percentile_scalar_s7s4'][i])
                            for i in range(uinp.sheep['i_salep_percentile_scalar_s7s4'].shape[0])]) * uinp.sheep['i_salep_price_max_s7']).astype(dtype)
    ##Max price in grids (adj sav - overwrites the previous values)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sav['salep_max'], 5)
    ##Max price in grids (adj sam - scales the values)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sam['salep_max'])
    ##Scalar for weight impact across the grid (sat adjusted)
    weight_scalar_s7s5s6 = weight_pricescalar_s7s5s6
    ##Scalar for score impact across the grid (sat adjusted)
    score_scalar_s7s5s6 = score_pricescalar_s7s5s6
    ##price for the analysis
    grid_s7s5s6 = grid_max_s7[:,na,na] * weight_scalar_s7s5s6 * score_scalar_s7s5s6
    return grid_s7s5s6


def f1_salep_mob(weight_s7pg, scores_s7s6pg, cvlw_s7s5pg, cvscore_s7s6pg,
                grid_weightrange_s7s5pg, grid_scorerange_s7s6p5g, grid_priceslw_s7s5s6pg):
    '''A function to calculate the average price of the mob based on the average specifications in the mob.
    This is to represent that the distribution of weight & specification reduces the mob average price
    This representation allows valuing individual animal management and reducing the mob distribution.
    Note: if the distribution extends below the lower range of weight or score in the grid these animals have zero value (ncv)'''

    ##loop on s7 to reduce memory
    saleprice_mobaverage_s7pg = np.zeros_like(weight_s7pg)
    for s7 in range(weight_s7pg.shape[0]):
        ## Probability for each lw step in grid based on the mob average weight and the coefficient of variation (CV) of weight
        ### probability of being less than the upper value of the step (roll) - probability of less than the lower value of the step
        prob_lw_s5pg = np.maximum(0, fun.f_norm_cdf(np.roll(grid_weightrange_s7s5pg[s7,...], -1, axis = 0), weight_s7pg[s7,...], cvlw_s7s5pg[s7,...])
                              - fun.f_norm_cdf(grid_weightrange_s7s5pg[s7,...], weight_s7pg[s7,...], cvlw_s7s5pg[s7,...]))
        ## Probability for each score step in grid (fat score/CS) based on the mob average score and the CV of quality score
        prob_score_s6pg = np.maximum(0, fun.f_norm_cdf(np.roll(grid_scorerange_s7s6p5g[s7,...], -1, axis = 0), scores_s7s6pg[s7,...], cvscore_s7s6pg[s7,...])
                                 - fun.f_norm_cdf(grid_scorerange_s7s6p5g[s7,...], scores_s7s6pg[s7,...], cvscore_s7s6pg[s7,...]))
        ##Probability for each cell of grid (assuming that weight & score are independent allows multiplying weight and score probabilities)
        prob_grid_s5s6pg = prob_lw_s5pg[:,na, ...] * prob_score_s6pg

        ##Average price for the mob is the sum of the probabilities in each cell of the grid and the price in that cell
        saleprice_mobaverage_s7pg[s7,...] = np.sum(prob_grid_s5s6pg * grid_priceslw_s7s5s6pg[s7,...], axis = (0, 1))
    return saleprice_mobaverage_s7pg


def f_sale_value(cu0, cx, o_rc, o_ffcfw_pg, dressp_adj_yg, dresspercent_adj_s6pg,
                 dresspercent_adj_s7pg, grid_price_s7s5s6pg, month_scalar_s7pg,
                 month_discount_s7pg, price_type_s7pg, cvlw_s7s5pg, cvscore_s7s6pg,
                 grid_weightrange_s7s5pg, grid_scorerange_s7s6pg, age_end_p5g1, discount_age_s7pg,sale_cost_pc_s7pg,
                 sale_cost_hd_s7pg, mask_s7x_s7pg, sale_agemax_s7pg1, dtype=None):
    ##Calculate condition score from relative condition
    cs_pg = f1_condition_score(o_rc, cu0)
    ##Calculate fat score from relative condition
    fs_pg = f1_fat_score(o_rc, cu0)
    ##Combine the scores into single array
    scores_s8p = np.stack([fs_pg, cs_pg], axis=0)
    ##Select the quality scores (s8) for each price grid (s7)
    scores_s7s6pg = scores_s8p[uinp.sheep['ia_s8_s7']][:,na,...]
    ##Dressing percentage to adjust price grid from $/kg DW to $/kg LW
    ### It is easier to convert the price to $/kg LW than it is to convert a distribution of LW and fat score to a distribution of dressed weight and fat score
    ### because dressing percentage changes with fat score.
    dresspercent_for_price_s7s6pg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[23, ...] + dresspercent_adj_s6pg + dresspercent_adj_s7pg[:,na,...]
    ##Dressing percentage is set to 100% if price type is $/kg LW or $/hd
    dresspercent_for_price_s7s6pg = fun.f_update(dresspercent_for_price_s7s6pg, 1, price_type_s7pg[:,na,...] >= 1)
    ##Create the grid prices in $/kg LW
    grid_priceslw_s7s5s6pg = grid_price_s7s5s6pg * dresspercent_for_price_s7s6pg[:,na,...]

    ## Calculate the 'lookup' weight of the average animal in the units of each grid (some grids the weight is dressed weight other grids are LW)
    ## start with dressing percentage and set to 1 later if the grid is kg LW
    ###Interploate DP adjustment based on the average FS of the animals
    dressp_adj_fs_pg= np.interp(fs_pg, uinp.sheep['i_salep_score_range_s8s6'][0, ...], uinp.sheep['i_salep_dressp_adj_s6']).astype(dtype)
    ###Average Dressing percentage including effects of genotype, fat score and age (which varies with the grid).
    dresspercent_for_wt_s7pg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[23, ...] + dressp_adj_fs_pg + dresspercent_adj_s7pg
    ###Dressing percentage is 100% if price type is $/kg LW or $/hd
    dresspercent_wt_s7pg = fun.f_update(dresspercent_for_wt_s7pg, 1, price_type_s7pg >= 1)
    ###Scale ffcfw to the units in the grid
    weight_for_lookup_s7pg = o_ffcfw_pg * dresspercent_wt_s7pg

    ##Calculate mob average price in each grid from the mob average and the distribution of weight & score within the mob (this is just the price, not the total animal value)
    price_mobaverage_s7pg = f1_salep_mob(weight_for_lookup_s7pg, scores_s7s6pg, cvlw_s7s5pg, cvscore_s7s6pg,
                                                      grid_weightrange_s7s5pg, grid_scorerange_s7s6pg, grid_priceslw_s7s5s6pg)

    ##Scale price received based on month of sale
    price_mobaverage_s7pg = price_mobaverage_s7pg * (1+month_scalar_s7pg)

    ## Apply the age based discount if the animal is greater than the threshold age
    ### Temporary value of the age based discount from the relevant month
    temporary_s7pg = price_mobaverage_s7pg * (1 + month_discount_s7pg)
    ###Apply discount if age is greater than threshold age
    price_mobaverage_s7pg = fun.f_update(price_mobaverage_s7pg, temporary_s7pg, age_end_p5g1/30 > discount_age_s7pg)  #divide 30 to convert to months

    ## Some grids are in $/hd. For these grids don't want to multiply grid value by weight (so set weight to 1)
    ### Convert weight to 1 if price is $/hd (price_type == 2)
    weight_for_value_s7pg = fun.f_update(o_ffcfw_pg, 1, price_type_s7pg == 2)

    ## Calculate the net value per head from the gross value minus the selling costs
    ### Calculate gross value per head
    sale_value_s7pg = price_mobaverage_s7pg * weight_for_value_s7pg
    ###Subtract the selling costs (some are percentage costs some are $/hd)
    sale_value_s7pg = sale_value_s7pg * (1 - sale_cost_pc_s7pg) - sale_cost_hd_s7pg

    ## Select the best net sale price from the relevant grids
    ###Mask the grids based on the maximum age and the gender for each grid
    sale_value_s7pg = sale_value_s7pg * mask_s7x_s7pg * (age_end_p5g1/30 <= sale_agemax_s7pg1) #divide 30 to convert to months
    ###Select the maximum value across the grids
    sale_value = np.max(sale_value_s7pg, axis=0) #take max on s6 axis as well to remove it (it is singleton so no effect)
    return sale_value

def f1_animal_trigger_levels(index_pg, age_start, period_is_shearing_pg, period_is_wean_pg, gender, o_ebg_p, wool_genes,
                            period_is_joining_pg, animal_mated, scan_option, period_is_endmating_pg):
    ##Trigger value 1 - week of year
    trigger1_pg = index_pg % 52
    ##Trigger value 2 - age
    trigger2_pg = np.trunc(age_start / 7)
    ##Trigger value 3 - Weeks from previous shearing
    trigger3_pg = index_pg - np.maximum.accumulate(index_pg*period_is_shearing_pg, axis=sinp.stock['i_p_pos'])
    ##Trigger value 4 - weeks to next shearing
    shear_idx = index_pg * period_is_shearing_pg
    shear_idx[np.logical_not(period_is_shearing_pg)] = np.max(index_pg) * 2 #set index to large number if not shearing
    trigger4_pg = np.flip(np.minimum.accumulate(np.flip(shear_idx, axis=sinp.stock['i_p_pos']), axis=sinp.stock['i_p_pos']), axis=sinp.stock['i_p_pos']) - index_pg
    ##Trigger value 5 - weeks from previous joining
    trigger5_pg = index_pg - np.maximum.accumulate(index_pg*period_is_joining_pg)
    ##Trigger value 6 - weeks from end of mating
    trigger6_pg = index_pg - np.maximum.accumulate(index_pg*period_is_endmating_pg)
    ##Trigger value 7 - weeks from previous weaning
    trigger7_pg = index_pg - np.maximum.accumulate(index_pg*period_is_wean_pg)
    ##Trigger value 8 - whether animals was mated
    trigger8_pg = animal_mated
    ##Trigger value 9 - scanning option being used
    trigger9_pg = scan_option
    ##Trigger value 10 - gender of the animal
    trigger10_pg = gender
    ##Trigger value 11 - rate of empty body gain
    trigger11_pg = o_ebg_p
    ##Trigger value 12 - the 'wooliness' of the genotype
    trigger12_pg = wool_genes
    ##Stack the triggers
    animal_triggervalues_h7pg = np.stack(np.broadcast_arrays(trigger1_pg, trigger2_pg, trigger3_pg, trigger4_pg,
                                                             trigger5_pg, trigger6_pg, trigger7_pg, trigger8_pg, trigger9_pg,
                                                             trigger10_pg, trigger11_pg, trigger12_pg), axis = 0)
    return animal_triggervalues_h7pg


def f_treatment_unit_numbers(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, a_nyatf_b1g=0):
    ##Unit 0 - per head
    unit0_pg = 1
    ##Unit 1 - adjusted head
    unit1_pg = unit0_pg * head_adjust
    ##Unit 2 - mob
    unit2_pg = unit0_pg / mobsize_pg
    ##Unit 3 - LW
    unit3_pg = unit0_pg * o_ffcfw_pg
    ##Unit 4 - CFW
    unit4_pg = unit0_pg * o_cfw_pg
    ##Unit 5 - nyatf
    unit5_pg = unit0_pg * a_nyatf_b1g
    ##Stack the triggers
    treatment_units_h8pg = np.stack(np.broadcast_arrays(unit0_pg, unit1_pg, unit2_pg, unit3_pg, unit4_pg, unit5_pg), axis=0)
    return treatment_units_h8pg


def f1_operations_triggered(animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg):
    shape = (operations_triggerlevels_h5h7h2pg.shape[2],) + animal_triggervalues_h7pg.shape[1:]
    triggered_h2pg = np.zeros(shape, dtype=bool)
    for h2 in range(operations_triggerlevels_h5h7h2pg.shape[2]):
        ##Test slice 0 of h5 axis
        slice0_h7pg = animal_triggervalues_h7pg[:, ...] <= operations_triggerlevels_h5h7h2pg[0, :, h2, ...]
        ##Test slice 1 of h5 axis
        slice1_h7pg = np.logical_or(animal_triggervalues_h7pg[:, ...] == operations_triggerlevels_h5h7h2pg[1, :, h2, ...],
                                    operations_triggerlevels_h5h7h2pg[1, :, h2, ...] == np.inf)
        ##Test slice 2 of h5 axis
        slice2_h7pg = animal_triggervalues_h7pg[:, ...] >= operations_triggerlevels_h5h7h2pg[2, :, h2, ...]
        ##Test across the conditions
        slices_all_h7pg = np.logical_and(slice0_h7pg, np.logical_and(slice1_h7pg, slice2_h7pg))
        ##Test across the rules (& collapse s7 axis)
        triggered_h2pg[h2,...] = np.all(slices_all_h7pg, axis=0)
    return triggered_h2pg


def f1_application_level(operation_triggered_h2pg, animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg):
    ##loop on h2 axis to save memory
    level_h2pg = np.ones_like(operation_triggered_h2pg, dtype='float32')
    for h2 in range(operation_triggered_h2pg.shape[0]):

        ## mask & remove the slices of the h7 axis that don't require calculation of the application level (not required because inputs do not include a range input)
        ## must be same mask for 'le' and 'ge'
        maskh7_h7 = fun.f_reduce_skipfew(np.any, operations_triggerlevels_h5h7h2pg[3,:, h2, ...] != np.inf, preserveAxis=0)

        ##slice operation_triggered array
        operation_triggered_pg = operation_triggered_h2pg[h2,...]

        ##if all values in mask are false (eg no range level needs to be calculated) then skip to next h2 (final array has 1 as default value so nothing needs to happen)
        if any(maskh7_h7):
            ### mask the input arrays to minimise slices of h7
            animal_triggervalues_h7mask_h7pg = animal_triggervalues_h7pg[maskh7_h7]
            operations_triggerlevels_h7mask_h5h7pg = operations_triggerlevels_h5h7h2pg[:, maskh7_h7, h2, ...]


            ##broadcast the input arrays so the 'required' mask can be applied
            operations_triggerlevels_casted_h5h7pg=np.broadcast_to(operations_triggerlevels_h7mask_h5h7pg, operations_triggerlevels_h7mask_h5h7pg.shape[0:2]+operation_triggered_pg.shape)
            animal_triggervalues_h7mask_h7pg = np.broadcast_to(animal_triggervalues_h7mask_h7pg, operations_triggerlevels_casted_h5h7pg.shape[1:])


            ## Calculate the application level for "less than or equal"
            ### The 'le' calculation is required only if the 'range' input is less than the le trigger value and both are not inf.
            required_h7pg = (operation_triggered_pg * (operations_triggerlevels_h7mask_h5h7pg[0, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] < operations_triggerlevels_h7mask_h5h7pg[0, ...]))

            ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
            temporary_h7pg = np.ones_like(required_h7pg, dtype='float32')

            ##Level if animal trigger level is between 'range' and 'le'
            ### calculate the masked version of the triggerlevels because required 3 times in the calculation
            operations_triggerlevels_masked_h5h7pg = operations_triggerlevels_casted_h5h7pg[:, required_h7pg]
            temporary_h7pg[required_h7pg] = np.clip((animal_triggervalues_h7mask_h7pg[required_h7pg] - operations_triggerlevels_masked_h5h7pg[0, ...])/
                                                        (operations_triggerlevels_masked_h5h7pg[3, ...] - operations_triggerlevels_masked_h5h7pg[0, ...]),0,1)
            ##Select the maximum across the h7 axis if the operation is triggered
            level_pg = np.max(temporary_h7pg,axis=0) * operation_triggered_pg   #mul by operation triggered so that level goes to 0 if operation is not triggered


            ## Repeat for 'ge' using same variable names as for 'le'
            ## Calculate the application level for "greater than or equal"
            ### The 'ge' calculation is required only if the 'range' input is greater than the ge trigger value and both are not inf.
            required_h7pg = (operation_triggered_pg * (operations_triggerlevels_h7mask_h5h7pg[2, ...] != -np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] > operations_triggerlevels_h7mask_h5h7pg[2, ...]))

            ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
            ### calculate the masked version of the triggerlevels because required 3 times in the calculation
            operations_triggerlevels_masked_h5h7pg = operations_triggerlevels_casted_h5h7pg[:, required_h7pg]
            temporary_h7pg = np.ones_like(required_h7pg, dtype='float32')

            ##Level if animal trigger level is between 'range' and 'le'
            temporary_h7pg[required_h7pg] = np.clip((animal_triggervalues_h7mask_h7pg[required_h7pg] - operations_triggerlevels_masked_h5h7pg[2, ...])/
                                                        (operations_triggerlevels_masked_h5h7pg[3, ...] - operations_triggerlevels_masked_h5h7pg[2, ...]),0,1)
            ##Select the maximum across the h7 axis if the operation is triggered
            temporary_pg = np.max(temporary_h7pg,axis=0) * operation_triggered_pg   #mul by operation triggered so that level goes to 0 if operation is not triggered

            ##Select the maximum of the 'le' and 'ge' value
            level_h2pg[h2,...] = np.maximum(level_pg, temporary_pg)
        ##if there is no range then level is just 1 * triggered
        else:
            level_h2pg[h2,...] = level_h2pg[h2,...] * operation_triggered_pg

    return level_h2pg


def f1_mustering_required(application_level_h2pg, husb_operations_muster_propn_h2pg):
    ##Total mustering required for all operations
    musters_pg = np.sum(application_level_h2pg * husb_operations_muster_propn_h2pg, axis=0)
    ##Round up to the next integer
    musters_pg = np.ceil(musters_pg)
    return musters_pg


# def f_husbandry_component(level, treatment_units, requirements, association, axes_tup):
#     ##Number of treatment units for contract
#     units = treatment_units[association]
#     ##Infrastructure requirement for each animal class during the period
#     component = np.sum(level * units * requirements, axis=axes_tup)
#     return component


def f1_husbandry_requisites(level_hpg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_requisites_prob_h6hpg,a_h8_h):
    ##Number of treatment units for requisites
    if type(a_h8_h)==int:
        units_hpg = treatment_units_h8pg[a_h8_h:a_h8_h+1] #so the h axis is kept
    else:
        units_hpg = treatment_units_h8pg[a_h8_h]
    ##Labour requirement for each animal class during the period
    ##calculated using loop to reduce memory
    cost_pg = 0
    for h in range(level_hpg.shape[0]):
        cost_pg += np.sum(level_hpg[h] * units_hpg[h] * husb_requisite_cost_h6pg *
                     husb_requisites_prob_h6hpg[:,h], axis = 0)
    return cost_pg


def f1_husbandry_labour(level_hpg, treatment_units_h8pg, units_per_labourhour_l2hpg, a_h8_h):
    ##Number of treatment units for contract
    if type(a_h8_h)==int:
        units_hpg = treatment_units_h8pg[a_h8_h:a_h8_h+1] #so the h axis is kept
    else:
        units_hpg = treatment_units_h8pg[a_h8_h]
    ##Labour requirement for each animal class during the period
    ##calculated using loop to reduce memory
    hours_l2pg = 0
    for h2 in range(level_hpg.shape[0]):
        hours_l2pg += fun.f_divide(level_hpg[h2] * units_hpg[h2] , units_per_labourhour_l2hpg[:,h2], dtype=level_hpg.dtype) #divide by units_per_labourhour_l2hpg because that is how many units can be done per hour eg how many sheep can be drenched per hr
    return hours_l2pg


def f1_husbandry_infrastructure(level_hpg, husb_infrastructurereq_h1h2pg):
    ##Infrastructure requirement for each animal class during the period
    ##calculated using loop to reduce memory
    infrastructure_h1pg = 0
    for h2 in range(level_hpg.shape[0]):
        infrastructure_h1pg += level_hpg[h2] * husb_infrastructurereq_h1h2pg[:,h2]
    return infrastructure_h1pg


def f1_contract_cost(application_level_h2pg, treatment_units_h8pg, husb_operations_contract_cost_h2pg):
    ##Number of animal units for contract
    units_h2pg = treatment_units_h8pg[uinp.sheep['ia_h8_h2']]
    ##Contract cost for each animal class during the period
    cost_pg = np.sum(application_level_h2pg * units_h2pg * husb_operations_contract_cost_h2pg, axis=0)
    return cost_pg


def f_husbandry(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, operations_triggerlevels_h5h7h2pg, index_pg,
                age_start, period_is_shear_pg, period_is_wean_pg, gender, o_ebg_p, wool_genes,
                husb_operations_muster_propn_h2pg, husb_requisite_cost_h6pg, husb_operations_requisites_prob_h6h2pg,
                operations_per_hour_l2h2pg, husb_operations_infrastructurereq_h1h2pg,
                husb_operations_contract_cost_h2pg, husb_muster_requisites_prob_h6h4pg,
                musters_per_hour_l2h4pg, husb_muster_infrastructurereq_h1h4pg,
                a_nyatf_b1g=0,period_is_joining_pg=False, animal_mated=False, scan_option=0, period_is_endmating_pg=False, dtype=None):
    ##An array of the trigger values for the animal classes in each period - these values are compared against a threshold to determine if the husb is required
    animal_triggervalues_h7pg = f1_animal_trigger_levels(index_pg, age_start, period_is_shear_pg, period_is_wean_pg, gender,
                            o_ebg_p, wool_genes, period_is_joining_pg, animal_mated, scan_option, period_is_endmating_pg).astype(dtype)
    ##The number of treatment units per animal in each period - each slice has a different unit eg mobsize, nyatf etc the treatment unit can be selected and applied for a given husb operation
    treatment_units_h8pg = f_treatment_unit_numbers(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, a_nyatf_b1g).astype(dtype)
    ##Is the husb operation triggered in the period for each class
    operation_triggered_h2pg = f1_operations_triggered(animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg)
    ##The level of the operation in each period for the class of livestock (proportion of animals that receive treatment) - this accounts for the fact that just because the operation is triggered the operation may not be done to all animals
    application_level_h2pg = f1_application_level(operation_triggered_h2pg, animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg)
    ##The number of times the mob must be mustered
    mustering_level_h4pg = f1_mustering_required(application_level_h2pg, husb_operations_muster_propn_h2pg)[na,...] #needs a h4 axis for the functions below
    ##The cost of requisites for the operations
    operations_requisites_cost_pg = f1_husbandry_requisites(application_level_h2pg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_operations_requisites_prob_h6h2pg, uinp.sheep['ia_h8_h2'])
    ##The labour requirement for the operations
    operations_labourreq_l2pg = f1_husbandry_labour(application_level_h2pg, treatment_units_h8pg, operations_per_hour_l2h2pg, uinp.sheep['ia_h8_h2'])
    ##The infrastructure requirements for the operations
    operations_infrastructurereq_h1pg = f1_husbandry_infrastructure(application_level_h2pg, husb_operations_infrastructurereq_h1h2pg)
    ##Contract cost for husbandry
    contract_cost_pg = f1_contract_cost(application_level_h2pg, treatment_units_h8pg, husb_operations_contract_cost_h2pg)
    ##The cost of requisites for mustering
    mustering_requisites_cost_pg = f1_husbandry_requisites(mustering_level_h4pg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_muster_requisites_prob_h6h4pg, uinp.sheep['ia_h8_h4'])
    ##The labour requirement for mustering
    mustering_labourreq_l2pg = f1_husbandry_labour(mustering_level_h4pg, treatment_units_h8pg, musters_per_hour_l2h4pg, uinp.sheep['ia_h8_h4'])
    ##The infrastructure requirements for mustering
    mustering_infrastructurereq_h1pg = f1_husbandry_infrastructure(mustering_level_h4pg, husb_muster_infrastructurereq_h1h4pg)
    ##Total cost of husbandry
    husbandry_cost_pg = operations_requisites_cost_pg + mustering_requisites_cost_pg + contract_cost_pg
    ##Labour requirement for husbandry
    husbandry_labour_l2pg = operations_labourreq_l2pg + mustering_labourreq_l2pg
    ##infrastructure requirement for husbandry
    husbandry_infrastructure_h1pg = operations_infrastructurereq_h1pg + mustering_infrastructurereq_h1pg
    return husbandry_cost_pg, husbandry_labour_l2pg, husbandry_infrastructure_h1pg


##################
#post processing #
##################

##Method 1 (still used)- add p and v axis together then sum p axis - this may be a good method for faster computers with more memory
def f1_p2v_std(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
            period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
    ## convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
    try:
        days_period_p = days_period_p.astype('float32')
    except AttributeError:
        pass
    ##mul everything
    production_ftvpany = (production_p * numbers_p * days_period_p * period_is_tvp
                          * on_hand_tvp * (dvp_pointer_p == index_vp) * (a_any1_p == index_any1tvp)
                          * (a_any2_p == index_any2any1tvp))
    ## sum along p axis to leave just a v axis (sumadj is to handle nsire that has a p8 axis at the end)
    return np.sum(production_ftvpany, axis=sinp.stock['i_p_pos']-sumadj)


##Method 2 (fastest)- sum sections of p axis to leave v (almost like sum if) this is fast because don't need p and v axis in same array
def f1_p2v(production_p, dvp_pointer_p=1, numbers_p=1, on_hand_tp=True, days_period_p=1, period_is_tp=True, a_any1_p=1, index_any1tp=1, a_any2_p=1, index_any2any1tp=1):
    #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
    try:
        days_period_p = days_period_p.astype('float32')
    except AttributeError:
        pass
    ##mul everything - add t,f and p6 axis
    production_ftpany = (production_p * numbers_p * days_period_p * period_is_tp
                        * on_hand_tp * (a_any1_p==index_any1tp)
                        * (a_any2_p==index_any2any1tp))
    ##convert p to v - info at this link https://stackoverflow.com/questions/50121980/numpy-conditional-sum
    ##basically we are summing the p axis for each dvp. the tricky part (which has caused the requirement for the loops) is that dvp pointer is not the same for each axis eg dvp is effected by e axis.
    ##so we need to loop though all the axis in the dvp and sum p and assign to a final array.
    ##if the axis is size 1 (ie singleton) then we want to take all of that axis ie ':' because just because the dvp pointer has singleton doesnt mean param array has singleton so need to take all slice of the param (unless that is an active dvp axis because that means dvp timing may differ for different slices along that axis so it must be summed in the loop)
    shape = production_ftpany.shape[0:sinp.stock['i_p_pos']] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[sinp.stock['i_p_pos']+1:]  # bit messy because need v t and all the other axis (but not p)
    result=np.zeros(shape).astype('float32')
    shape = dvp_pointer_p.shape
    #todo referring to axes positions using a constant rather than using the variable
    for a1 in range(shape[-14]):
        a1_slc = slice(a1,a1+1) if shape[-14]>1 else slice(0,None) #used for param because we want to keep axis
        for e1 in range(shape[-13]):
            e1_slc = slice(e1, e1 + 1) if shape[-13] > 1 else slice(0, None)
            for b1 in range(shape[-12]):
                b1_slc = slice(b1, b1 + 1) if shape[-12] > 1 else slice(0, None)
                for n in range(shape[-11]):
                    n_slc = slice(n, n + 1) if shape[-11] > 1 else slice(0, None)
                    for w in range(shape[-10]):
                        w_slc = slice(w, w + 1) if shape[-10] > 1 else slice(0, None)
                        for z in range(shape[-9]):
                            z_slc = slice(z, z + 1) if shape[-9] > 1 else slice(0, None)
                            for i in range(shape[-8]):
                                i_slc = slice(i, i + 1) if shape[-8] > 1 else slice(0, None)
                                for d in range(shape[-7]):
                                    d_slc = slice(d, d + 1) if shape[-7] > 1 else slice(0, None)
                                    for a0 in range(shape[-6]):
                                        a0_slc = slice(a0, a0 + 1) if shape[-6] > 1 else slice(0, None)
                                        for e0 in range(shape[-5]):
                                            e0_slc = slice(e0, e0 + 1) if shape[-5] > 1 else slice(0, None)
                                            for b0 in range(shape[-4]):
                                                b0_slc = slice(b0, b0 + 1) if shape[-4] > 1 else slice(0, None)
                                                for x in range(shape[-3]):
                                                    x_slc = slice(x, x + 1) if shape[-3] > 1 else slice(0, None)
                                                    for y in range(shape[-2]):
                                                        y_slc = slice(y, y + 1) if shape[-2] > 1 else slice(0, None)
                                                        for g in range(shape[-1]):
                                                            g_slc = slice(g, g + 1) if shape[-1] > 1 else slice(0, None)
                                                            result[..., a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc] \
                                                                = np.add.reduceat(production_ftpany[..., a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                                  , np.r_[0, np.where(np.diff(dvp_pointer_p[:, a1, e1, b1, n, w, z, i, d, a0, e0, b0, x, y, g]))[0] + 1], axis=sinp.stock['i_p_pos']) #np.r_ basically concats two 1d arrays (so here we are just adding 0 to the start of the array)
    return result

# ##Method 5 numexpr (slow - even with fv33n33 it is much slower than the current method
# import numexpr as ne
# def f1_p2v_5(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
#             period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
#     ## convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     try:
#         days_period_p = days_period_p.astype('float32')
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_any1_p==index_any1tvp)
#                         * (a_any2_p==index_any2any1tvp))* (dvp_pointer_p == index_vp)
#     ##mul everything and sum - both methods below are slow. but the least amount of calc in the ne.eval the better.
#     production_ftvany = ne.evaluate("sum(production_ftpany,axis=5)")
#     # production_ftvany = ne.evaluate("sum(production_p * numbers_p * days_period_p * period_is_tvp * on_hand_tvp * (dvp_pointer_p == index_vp) * (a_any1_p == index_any1tvp) * (a_any2_p == index_any2any1tvp),axis=5)")
#
#     return production_ftvany


# ##Method 4 - loop over v and sum p - this save p and v axis being on the same array but requires lots of looping so isn't much faster
# def f1_p2v_loop(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1, period_is_tvp=True, a_ev_p=1, index_ftvp=1, a_p6_p=1, index_p6ftvp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_ev_p==index_ftvp)
#                         * (a_p6_p==index_p6ftvp))
#
#     shape = production_ftpany.shape[0:3] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[4:]  # bit messy because need v t and all the other axis (but not p)
#     final=np.zeros(shape).astype('float32')
#     for i in range(np.max(dvp_pointer_p)+1):
#         temp_prod = np.sum(production_ftpany * (dvp_pointer_p==i), axis=sinp.stock['i_p_pos'])
#         final[:,:,:,i,...] = temp_prod  #asign to correct v slice
#     return final

# ##Method 3 - use groupby to sum p, this means p and v don't exist on the same array - not as fast as method 2
# import numpy_indexed as npi
# def f1_p2v_groupby(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1, period_is_tvp=True, a_ev_p=1, index_ftvp=1, a_p6_p=1, index_p6ftvp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_ev_p==index_ftvp)
#                         * (a_p6_p==index_p6ftvp))
#     ##convert p to v
#     shape = production_ftpany.shape[0:3] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[4:]  # bit messy because need v t and all the other axis (but not p)
#     result=np.zeros(shape).astype('float32')
#     shape = dvp_pointer_p.shape
#     for e1 in range(shape[-13]):
#         for g in range(shape[-1]):
#             result[:,:,:,:, :, e1:e1+1, :, :, :, :, :, :, :, :, :, :, :, g:g+1] = npi.GroupBy(dvp_pointer_p[:, 0, e1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g], axis=0).sum(
#                                                                     production_ftpany[:, :, :, :, :, e1:e1+1, :, :, :, :, :, :, :, :, :, :, :, g:g+1], axis=sinp.stock['i_p_pos'])[1]
#     return result


def f1_cum_dvp(arr,dvp_pointer,axis=0,shift=0):
    '''This function does accumulative max but it resets at each dvp.
    '''
    final = np.zeros_like(arr)
    for i in range(np.max(dvp_pointer)+1):  #plus 1 so that the last dvp is counted for
        arr1 = arr * (dvp_pointer==i) #sets the p slices to 0 if not in the given dvp
        arr1 = np.roll(arr1,shift,axis) #this is only used for the dams on hand calculation, this rolls the period is sale array 1 unit along the p axis.
                                        # This is required so that period is onhand == true in the period that sale occurs and false after that.
                                        # Because sale occurs at the end of a given period so the sheep are technically onhand for the period sale occurs.
        arr1 = np.maximum.accumulate(arr1,axis=axis)
        arr1 = arr1 * (dvp_pointer==i) #sets the cum max to 0 for other dvp not of interest
        final += arr1
    return final


def f1_cum_sum_dvp(arr,dvp_pointer,axis=0,shift=0):
    '''This function does accumulative sum but it resets at each dvp.
    '''
    final = np.zeros_like(arr)
    for i in range(np.max(dvp_pointer)+1):  #plus 1 so that the last dvp is counted for
        arr1 = arr * (dvp_pointer==i) #sets the p slices to 0 if not in the given dvp
        arr1 = np.roll(arr1,shift,axis) #this is only used for the dams on hand calculation, this rolls the period is sale array 1 unit along the p axis.
                                        # This is required so that period is onhand == true in the period that sale occurs and false after that.
                                        # Because sale occurs at the end of a given period so the sheep are technically onhand for the period sale occurs.
        arr1 = np.cumsum(arr1,axis=axis)
        arr1 = arr1 * (dvp_pointer==i) #sets the cum max to 0 for other dvp not of interest
        final += arr1
    return final


def f1_lw_distribution(ffcfw_dest_w8g, ffcfw_source_w8g, index_w8=None, dvp_type_next_tvgw=0, vtype=0): #, w_pos, i_n_len, i_n_fvp_period, dvp_type_next_tvgw=0, vtype=0):
    '''distributing animals on LW at the start of dvp
        the 8 or 9 is dropped from the w if singleton'''
    ## Move w axis of dest_w8g to -1 and f_expand to retain the original ‘w’ as a singleton
    ffcfw_dest_wgw9 = fun.f_expand(np.moveaxis(ffcfw_dest_w8g, sinp.stock['i_w_pos'],-1), sinp.stock['i_n_pos']-1, right_pos=sinp.stock['i_z_pos']-1)
    ## create index for w9 based on shape of the (now) last axis (to be used later)
    index_w9 = np.arange(ffcfw_dest_wgw9.shape[-1])

    ## Create the distribution (result) array because it is assigned in slices
    distribution_nearest_w8gw9 = np.zeros_like(np.broadcast_arrays(ffcfw_source_w8g[...,na],ffcfw_dest_wgw9)[0]) #broadcast array returns list of two arrays so need to select one
    distribution_nextnearest_w8gw9 = np.zeros_like(distribution_nearest_w8gw9)

    ## Find the index of the destination slice that is nearest to the source weight
    diff_w8gw9 = ffcfw_dest_wgw9 - ffcfw_source_w8g[...,na]
    diff_abs_w8gw9 = np.abs(diff_w8gw9)
    nearestw9_idx_w8g = np.argmin(diff_abs_w8gw9,axis = -1)

    ## If an index_w8 has been provided (because it is a square w8:w9) then test for equality
    ### if the source weight is matched to the destination then set index to own slice (so slice distributes to itself)
    #### required if destination weights are replicated and nearestw9_idx is pointing to first occurrence
    if index_w8 is not None:
        nearestw9_idx_w8g = fun.f_update(nearestw9_idx_w8g, index_w8, np.isclose(ffcfw_dest_w8g, ffcfw_source_w8g))

    ## The nearest destination weight for each source weight & the difference from each w8
    nearestw9_w8gw = np.take_along_axis(ffcfw_dest_wgw9, nearestw9_idx_w8g[...,na], axis=-1)
    diff_nearest_w8gw = np.take_along_axis(diff_w8gw9, nearestw9_idx_w8g[...,na], axis=-1)  #alternate calc: diff_nearest_w8gw = nearestw9_w8gw - ffcfw_source_w8g[...,na]

    ## Determine the index of the next nearest destination slice that is on the opposite side of the source (using masked array)
    ### mask the values for which the difference is the same sign as the difference of the nearest.
    mask = np.sign(diff_w8gw9) == np.sign(diff_nearest_w8gw)
    next_nearestw9_idx_w8g = np.argmin(np.ma.masked_array(diff_abs_w8gw9, mask), axis = -1)

    ## If an index_w8 has been provided then test for equality (as per nearest)
    if index_w8 is not None:
        next_nearestw9_idx_w8g = fun.f_update(next_nearestw9_idx_w8g, index_w8, np.isclose(ffcfw_dest_w8g, ffcfw_source_w8g))

    ## the next_nearest destination weight
    next_nearestw9_w8gw = np.take_along_axis(ffcfw_dest_wgw9, next_nearestw9_idx_w8g[...,na], axis=-1)

    ## Calculate the proportion distributed to the nearest and assign to that w9 slice
    ### Handle the special cases in f_divide (option=1) where source and destination weights are the same
    ### weights have converged or the dest and source weight is 0 for all slices (eg if animals don't exist or distribution doesnt occur in the dvp)
    #### nearest
    proportion = fun.f_divide(ffcfw_source_w8g[...,na] - next_nearestw9_w8gw
                              , nearestw9_w8gw - next_nearestw9_w8gw, option=1)
    # handle situation when the destination weights are replicated but source is not (not sure that this can occur)
    proportion = fun.f_update(proportion, 1, np.isclose(nearestw9_w8gw, next_nearestw9_w8gw))
    np.put_along_axis(distribution_nearest_w8gw9, nearestw9_idx_w8g[...,na], proportion, axis=-1)
    #### next nearest
    proportion = fun.f_divide(nearestw9_w8gw - ffcfw_source_w8g[...,na]
                              , nearestw9_w8gw - next_nearestw9_w8gw, option=1)
    np.put_along_axis(distribution_nextnearest_w8gw9, next_nearestw9_idx_w8g[...,na], proportion, axis=-1)

    ## Handle the special cases where source weight is less than the lowest destination weight
    ### the light animals are transferred such that total LW remains the same prior to and after the distribution.
    ### therefore the number of animals is reduced during the transfer by the ratio: source wt / lowest destination wt.
    ### to transfer the full number of the light animals the minimum destination weight will need to be altered.
    ratio_w8gw = fun.f_divide(ffcfw_source_w8g, np.min(ffcfw_dest_wgw9,axis=-1))[...,na]
    ### where the ratio is below 1 it is applied to the nearest w9 slice
    mask_w8gw9 = (ratio_w8gw < 1) * (nearestw9_idx_w8g[...,na] == index_w9)
    distribution_nearest_w8gw9 = fun.f_update(distribution_nearest_w8gw9, ratio_w8gw, mask_w8gw9)

    ## Combine the values into the return variable
    ### clip (0 to 1) to handle the special case where source weight > the maximum destination weight
    distribution_w8gw9 = np.clip(distribution_nearest_w8gw9 + distribution_nextnearest_w8gw9,0,1)
    # distribution_error = np.any(np.sum(distribution_w8gw9, axis=-1)>1)

    ##Set default for DVPs that don’t require distributing to 1 (these are masked later to remove those that are not required)
    distribution_w8gw9 = fun.f_update(distribution_w8gw9, 1, dvp_type_next_tvgw!=vtype)
    return distribution_w8gw9


def f1_create_production_param(group, production_vg, a_kcluster_vg_1=1, index_ktvg_1=1, a_kcluster_vg_2=1, index_kktvg_2=1, numbers_start_vg=1, mask_vg=True, pos_offset=0):
    '''Can convert total production to per animal production including impact of death if numbers have been included.
    Apply the k clustering and collapse the e, b & d axes
    If numbers_start are not included then only applies k clustering - this is usually done if production is already per head'''
    if group=='sire':
        return fun.f_divide(production_vg, numbers_start_vg, dtype=production_vg.dtype)
    elif group=='dams':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * mask_vg
                                  , axis = (sinp.stock['i_b1_pos']-pos_offset, sinp.stock['i_e1_pos']-pos_offset), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1),
                                     axis=(sinp.stock['i_b1_pos']-pos_offset, sinp.stock['i_e1_pos']-pos_offset), keepdims=True), dtype=production_vg.dtype)
    elif group=='offs':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2)
                                  , axis = (sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2),
                                     axis=(sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True), dtype=production_vg.dtype)
