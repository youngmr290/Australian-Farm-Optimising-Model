"""
Created on 5/2/21

module: structural module - contains all the structural inputs (only to be change by a core AFO developer)

This module has no SA

@author: young
"""

##python modules (CANT import propertyinputs)
import pickle as pkl
import pandas as pd
import numpy as np

import Functions as fun

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
# read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
import os.path

try:
    if os.path.getmtime("Structural.xlsx") > os.path.getmtime("pkl_structural.pkl"):
        inputs_from_pickle = False
    else:
        inputs_from_pickle = True
        print('Reading structural inputs from pickle',end=' ',flush=True)
except FileNotFoundError:
    inputs_from_pickle = False

filename = 'pkl_structural.pkl'
##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading structural inputs from Excel',end=' ',flush=True)
    with open(filename,"wb") as f:
        ##general
        general_inp = fun.xl_all_named_ranges("Structural.xlsx","General")
        pkl.dump(general_inp,f,protocol=pkl.HIGHEST_PROTOCOL)

        ##sheep inputs
        stock_inp = fun.xl_all_named_ranges('Structural.xlsx','Stock',numpy=True)
        pkl.dump(stock_inp,f,protocol=pkl.HIGHEST_PROTOCOL)


##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(filename,"rb") as f:
        general_inp = pkl.load(f)

        stock_inp = pkl.load(f)

print('- finished')

##reshape require inputs
###lengths
len_b0 = stock_inp['ia_ppk5_lsb0'].shape[-1]
len_b1 = stock_inp['ia_ppk2g1_rlsb1'].shape[-1]
len_l = stock_inp['i_len_l']
len_m = stock_inp['i_len_m']
len_s = stock_inp['i_len_s']
len_r = stock_inp['i_n_r1type']

###shapes
rlsb1 = (len_r,len_l,len_s,len_b1)
mlsb1 = (len_m,len_l,len_s,len_b1)
lsb0 = (len_l,len_s,len_b0)

###stock
stock_inp['ia_ppk2g1_rlsb1'] = np.reshape(stock_inp['ia_ppk2g1_rlsb1'],rlsb1)
stock_inp['ia_ppk5_lsb0'] = np.reshape(stock_inp['ia_ppk5_lsb0'],lsb0)
stock_inp['ia_k2_mlsb1'] = np.reshape(stock_inp['ia_k2_mlsb1'],mlsb1)



##copy inputs so there is an origional (before SA) version
general = general_inp.copy()
stock = stock_inp.copy()

##############
#phases      #
##############
phases = {}
##rotation phases and constraints read in from excel
phases['phases'] = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header= None, index_col = 0, engine='openpyxl').T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc



###############
#landuses     #
###############
'''
A1, E1 are special sets used in con2 - currently not used
Note
- A1 is also used in pasture functions to build the germ df, so it cant be deleted
- C is used in stubble module, createmodel & mach
- C1 is used just in pasture functions
- sets now include capitals - this shouldn't effect con1 but it makes building the germ df easier
'''
landuse = {}
##all pas2 includes cont pasture - used in reporting
landuse['All_pas']={'a', 'ar'
                , 's', 'sr'
                , 'm'
                , 'u', 'ur','uc'
                , 'x', 'xr','xc'
                , 'j', 't', 'jr', 'tr','tc','jc'
                }
##next set is used in pasture.py for germination and phase area
landuse['pasture_sets']={'annual': {'a', 'ar'
                                , 's', 'sr'
                                , 'm'}
                        ,'lucerne':{'u', 'uc', 'ur'
                                   , 'x', 'xc', 'xr'}
                        ,'tedera':{'j','jc', 't','tc', 'jr', 'tr'}
                       }
##G and C1 are just used in pas.py for germination ^can be removed when germination is calculated from sim
landuse['G']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'a', 'ar'
                , 's', 'sr'
                , 'm'
                , 'u', 'ur'
                , 'x', 'xr'
                , 'j', 't', 'jr', 'tr'
                , 'G', 'Y', 'E', 'N', 'P', 'OF'
                , 'A', 'AR'
                , 'S', 'SR'
                , 'M'
                , 'U'
                , 'X'
                , 'T', 'J'} #all landuses
landuse['C1']={'E', 'N', 'P', 'OF', 'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #had to create a separate set because don't want the capital in the crop set above as it is used to create pyomo set


landuse['All']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r', 'a', 'ar', 's', 'sr', 'm', 'u', 'uc', 'ur', 'x', 'xc', 'xr', 'j','jc', 't','tc', 'jr', 'tr'} #used in reporting and bounds
landuse['C']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #all crops, used in stubble and mach (not used for rotations)
landuse['Hay']={'h'} #all crops that produce hay - used in machpyomo/coremodel for hay con
##special sets used in crop sim
landuse['Ys'] = {'Y'}
landuse['As'] = {'A','a'}
landuse['JR'] = {'jr'}
landuse['TR'] = {'tr'}
landuse['UR'] = {'ur'}
landuse['XR'] = {'xr'}
landuse['PAS'] = {'A', 'AR', 'S', 'SR', 'M','T','J','U','X', 'tc', 'jc', 'uc', 'xc'}
##sets used in to build rotations
landuse['A']={'a', 'ar','s', 'sr', 'm'
                , 'A', 'AR'
                , 'S', 'SR'
                , 'M'} #annual
landuse['A1']={'a',  's', 'm'} #annual not resown - special set used in pasture germ and con2 when determining if a rotation provides a rotation because in yr1 we don't want ar to provide an A because we need to distinguish between them
landuse['AR']={'ar', 'AR'} #resown annual
landuse['E']={'E', 'E1', 'OF', 'b', 'h', 'o', 'of', 'w'} #cereals
landuse['E1']={'E', 'b', 'h', 'o', 'w'} #cereals
landuse['J']={'J', 'j', 'jr'} #tedera
landuse['M']={'m', 'M'} #manipulated pasture
landuse['N']={'N', 'z','r'} #canolas
landuse['OF']={'OF', 'of'} #oats fodder
landuse['P']={'P', 'f','i', 'k', 'l', 'v'} #pulses
landuse['S']={'s','sr', 'S', 'SR'} #spray topped pasture
landuse['SR']={'sr', 'SR'} #spray topped pasture
landuse['T']={'T', 't', 'tr','J', 'j', 'jr'} #tedera - also includes manipulated tedera because it is combined in yrs 3,4,5
landuse['U']={'u', 'ur', 'U','x', 'xr', 'X'} #lucerne
landuse['X']={'x', 'xr', 'X'} #lucerne
landuse['Y']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'Y', 'E', 'E1', 'N', 'P', 'OF'} #anything not pasture


'''make each landuse a set so the issuperset func works'''
landuse['a']={'a'}
landuse['ar']={'ar'}
landuse['b']={'b'}
landuse['f']={'f'}
landuse['h']={'h'}
landuse['i']={'i'}
landuse['j']={'j'}
landuse['jc']={'jc'}
landuse['jr']={'jr'}
landuse['k']={'k'}
landuse['l']={'l'}
landuse['m']={'m'}
landuse['o']={'o'}
landuse['of']={'of'}
landuse['r']={'r'}
landuse['s']={'s'}
landuse['sr']={'sr'}
landuse['t']={'t'}
landuse['tc']={'tc'}
landuse['tr']={'tr'}
landuse['u']={'u'}
landuse['uc']={'uc'}
landuse['ur']={'ur'}
landuse['v']={'v'}
landuse['w']={'w'}
landuse['x']={'x'}
landuse['xc']={'xc'}
landuse['xr']={'xr'}
landuse['z']={'z'}






#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
# functions that use data from above
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################


def end_col():
    #specify the column name/number for the current landuse
    end_col  = [general['phase_len']-1]
    return end_col