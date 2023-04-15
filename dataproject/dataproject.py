import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import pydst
dst = pydst.Dst(lang='da')

#creates dictionary for data
columns_dict = {}
columns_dict['AGEBYGROUP'] = 'agegroup'
columns_dict['MEDICINTYPE'] = 'medicinetype'
columns_dict['TID'] = 'year'
columns_dict['INDHOLD'] = 'count'
columns_dict['Bnøgle'] = 'unit'

var_dict = {} # var is for variable
var_dict['N05 Psycholeptica'] = '5'
var_dict['N06 Psychoanaleptica'] = '6'

unit_dict = {}
unit_dict['Personer'] = 'person'
unit_dict['Indløste recepter'] = 'recepter'

#import data on use of medicine from dst
Medicin4_true = dst.get_data(table_id = 'MEDICIN4', variables={'TID':['*'], 'AGEBYGROUP':['*'], 
'MEDICINTYPE':['*']})

#removes all medicinetypes that isn't Psycholeptica (downer used for anxiety and OCD)


#drops unnecessary colums
Medicine_drop = Medicine.drop('BNØGLE',axis=1)
Medicine_clean = Medicine_drop.drop('KØN',axis=1)