import os
import pandas as pd
import numpy as np

from core.power import Power

# 统计离线负荷
base_path = 'C:/PSASP_Pro/2020国调年度'
index = []
loads_sum = []
for sub in os.listdir(base_path):
    path = os.path.join(base_path, sub)
    if not os.path.isdir(path):
        continue
    index.append(sub)
    power = Power('off')
    power.load_power(path, 'off', lf=True, lp=False, st=False, station=False)
    power.data['load']['province'] = power.data['load']['name'].str[0]
    loads = power.data['load']
    # loads = loads[loads['province'].isin(['黑', '吉', '辽'])]
    loads = loads[loads['mark'] == 1]
    loads_sum.append(loads.groupby('province', sort=False).agg({'p0': 'sum'}))
    print(sub, 'loaded.')

loads_sum = pd.concat(loads_sum, axis=1).T
loads_sum.index = index

# 统计在线负荷
file_name = 'C:/Users/sdy/data/db/2019_09_12/data/station_pl.npz'
arch = np.load(file_name, allow_pickle=True)
loads = pd.DataFrame(arch['data'], index=arch['times'], columns=arch['elems'])
loads.fillna(0., inplace=True)
file_name = 'C:/Users/sdy/data/db/2019_09_12/db_2019_11_15T10_00_00/st_info.dat'
st_info = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=['name'])
names = st_info[st_info['type'] == 60].index
names = [n for n in names if n in loads.columns]
loads = loads[names]
loads_sum = []
for i in ['辽宁', '吉林', '黑龙江']:
    sub = loads.loc[:, loads.columns.str.contains(i)]
    sub = sub.clip(lower=0.)
    loads_sum.append(sub.sum(axis=1))
loads_sum = pd.concat(loads_sum, axis=1)
loads_sum.columns = ['辽宁', '吉林', '黑龙江']
loads_sum = loads_sum[~np.any(loads_sum == 0., axis=1)]


# 统计在线机组有功
file_name = 'C:/Users/sdy/data/db/2019_09_12/data/generator_p.npz'
arch = np.load(file_name, allow_pickle=True)
gens = pd.DataFrame(arch['data'], index=arch['times'], columns=arch['elems'])
gens.fillna(0., inplace=True)
file_name = 'C:/Users/sdy/data/db/2019_09_12/db_2019_11_15T10_00_00/elem_info.dat'
elem_info = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=['name'])
gen_info = elem_info[elem_info['type'] == 5]
gens_sum = []
for i in [21, 22, 23]:
    names = gen_info[gen_info['area'] == i].index
    names = [n for n in names if n in gens.columns]
    sub = gens[names]
    sub = sub.clip(lower=0.)
    gens_sum.append(sub.sum(axis=1))
gens_sum = pd.concat(gens_sum, axis=1)
gens_sum.columns = ['辽宁', '吉林', '黑龙江']
gens_sum = gens_sum[~np.any(gens_sum == 0., axis=1)]
