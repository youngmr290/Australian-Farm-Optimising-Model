"""

This module reads in rotation information that has been generated from RotGeneration.py and manipulates
it to produce the rotation parameters.

author: young
"""

#python modules
import pandas as pd
import numpy as np
import timeit


#AFO modules
import StructuralInputs as sinp
import PropertyInputs as pinp

def landuses_phases(params,report):
    '''
    * Read in the rotation list generated by RotGeneration.py
    * Create rotation area parameter for pyomo (simply each rotation phase uses 1ha of area).
    * Store rotation list and pasture phases list to report dictionary

    '''
    phases=sinp.f_phases()
    phases_rk = phases.set_index(5, append=True) #add landuse as index level
    params['phases_rk'] = dict.fromkeys(phases_rk.index,1)
    report['phases']=phases
    report['all_pastures']=sinp.landuse['All_pas'] #all_pas2 includes the cont pasture landuses


def rot_params(params):
    '''
    Create parameters for:

    * lmu area.
    * landuse history provided and required by each rotation phase.

    '''
    ##area
    lmu_mask = pinp.general['lmu_area'].squeeze() > 0
    params['lmu_area'] =  pinp.general['lmu_area'].squeeze()[lmu_mask].to_dict()

    #############################
    #rotation phase constraint1 #
    #############################    
    rot_req = pd.read_excel('Rotation.xlsx', sheet_name='rotation_req', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_prov = pd.read_excel('Rotation.xlsx', sheet_name='rotation_prov', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_req = rot_req.set_index([0,1])
    rot_prov = rot_prov.set_index([0,1])
    params['hist_prov'] = rot_prov.squeeze().to_dict()
    params['hist_req'] = rot_req.squeeze().to_dict()



