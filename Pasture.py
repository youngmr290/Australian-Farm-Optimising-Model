# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:15:40 2019

Version Control:
Version     Date        Person  Change
   1.1      30Dec19     JMY     Added timing to check elapased time for each function in pasture functions

Known problems:
Fixed   Date    ID by   Problem
        

@author: John
"""

import PastureFunctions as pfun
from timeit import default_timer as timer

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")


pfun.init_and_read_excel('Property.xlsx', 'annual')                         # read inputs from Excel file and map to the python variables
time_list.append(timer()) ; time_was.append("init & read inputs from Excel")

pfun.calculate_germ_and_reseeding()                          # calculate the germination for each rotation phase
time_list.append(timer()) ; time_was.append("germination")

pfun.green_and_dry()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("green feed")

poc_con_dict = pfun.poc_con()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc con")

poc_md_dict = pfun.poc_md()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc_md")

poc_vol_dict = pfun.poc_vol()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("poc_vol")




#report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in secondsfirst


#test times
#def test1():
#    annual.germ_phase_data.columns.values[range(phase_len)] = [*range(phase_len)]
#def test2():
#    annual.germ_phase_data.columns.values[0:phase_len] = [*range(phase_len)]
#    
#print(timeit.repeat(test1,number=5,repeat=10))
#print(timeit.repeat(test2,number=5,repeat=10))    
