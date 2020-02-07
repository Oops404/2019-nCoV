# -*- coding: UTF-8-*-
"""
@Project: 2019-nCoV
@Author: CheneyJin
@Time: 2020/2/1 13:18
"""

import json
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.integrate as spi
import numpy as np
from decimal import Decimal

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def read_json_from_file(path):
    with open(path, encoding='UTF-8') as fp:
        data = json.load(fp)
        return data


def read_whole():
    return read_json_from_file("./data/0.json")


def read_region():
    return read_json_from_file("./data/1.json")


whole_list = read_whole()['results']
region_list = read_region()['results']

whole_date_set = set()
whole_df = pd.DataFrame(columns=("confirmed", "suspected", "cured", "dead", "date"))
whole_index = 0
for daily in whole_list:
    time_local = time.localtime(daily["updateTime"] / 1000)
    date = time.strftime("%Y-%m-%d", time_local)
    format_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)

    if date not in whole_date_set:
        whole_df.loc[whole_index] = [daily["confirmedCount"], daily["suspectedCount"],
                                     daily["curedCount"], daily["deadCount"], date]
        whole_date_set.add(date)
        whole_index = whole_index + 1
# , ignore_index=True
whole_df = whole_df.sort_values("date", ascending=True, inplace=False, ignore_index=True)
# xticks=tuple(whole_df["date"])
whole_df.plot()
plt.show()

print(whole_df)

each_region_data_frame_dict = dict()
each_region_date_dict = dict()

region_index = 0
ignore_region = {"瑞典", "西班牙", "芬兰", "斯里兰卡", "柬埔寨", "尼泊尔", "俄罗斯",
                 "英国", "意大利", "法国", "菲律宾", "阿联酋", "待明确地区", "德国", "越南",
                 "比利时", "印度", "泰国", "加拿大", "马来西亚", "新加坡", "韩国", "美国", "日本", "澳大利亚"}
for daily in region_list:
    province = daily["provinceName"]
    if province in ignore_region:
        continue
    if province not in each_region_data_frame_dict.keys():
        region_data_frame = pd.DataFrame(columns=("name", "confirmed", "suspected", "cured", "dead", "date"))
    else:
        region_data_frame = each_region_data_frame_dict[province]

    if province not in each_region_date_dict.keys():
        province_date_set = set()
    else:
        province_date_set = each_region_date_dict[province]

    time_local = time.localtime(daily["updateTime"] / 1000)
    date = time.strftime("%Y-%m-%d", time_local)
    format_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)

    if date in province_date_set:
        continue

    region_data_frame.loc[region_index] = [province, daily["confirmedCount"],
                                           daily["suspectedCount"], daily["curedCount"],
                                           daily["deadCount"], format_time]
    province_date_set.add(date)
    each_region_data_frame_dict[province] = region_data_frame
    each_region_date_dict[province] = province_date_set
    region_index = region_index + 1


# for name in each_region_data_frame_dict:
#     frame = each_region_data_frame_dict[name]
#     frame = frame.sort_values("date", ascending=True, inplace=False, ignore_index=True)
#     frame.plot(title=name)
#     plt.show()


def cal_r0(confirmed, suspected, t):
    # confirmed是确诊人数；suspected是疑似人数；t是疾病已爆发时间

    # p为疑似病例转化为确诊病例的概率
    p = 0.695
    # Tg_1和Tg_2为生成时间（generation period）
    tg_1 = 8.4
    tg_2 = 10.0
    # yt为实际预估感染人数
    yt = suspected * p + confirmed
    # lamda为早期指数增长的增长率
    lamda = math.log(yt) / t

    r0_1 = 1 + lamda * tg_1 + p * (1 - p) * pow(lamda * tg_1, 2)
    r0_2 = 1 + lamda * tg_2 + p * (1 - p) * pow(lamda * tg_2, 2)

    return r0_1, r0_2


t_start = 23 + 24
r0_data_frame = pd.DataFrame(columns=("r0", "r1"))
r0_index = 0
for index, row in whole_df.iterrows():
    r01, r02 = cal_r0(row["confirmed"], row["suspected"], t_start)
    print("R0 -> {0}~{1}".format(Decimal(r01).quantize(Decimal("0.0000")), Decimal(r02).quantize(Decimal("0.0000"))))
    r0_data_frame.loc[r0_index] = [r01, r02]
    t_start = t_start + 1
    r0_index = r0_index + 1
r0_data_frame.plot(title="RO-MIN-MAX")
plt.show()
# print(each_region_data_frame_dict)

# gamma_1为潜伏期治愈率
gamma_1 = 0
# gamma_2为感染者治愈率
gamma_2 = 1188 / 24434  # 0.0235

# β为疾病传播概率
beta = 3.463173 * (1 / 14)  # 0.2561
# alpha为潜伏期发展为患者的比例
alpha = 1
# die为死亡率
die = 564 / 24434
# I_0为感染个体的初始比例
I_0 = 13 / 12000000
# E_0为潜伏期个体的初始比例
E_0 = I_0 * 51
# R_0为治愈个体的初始比例
R_0 = 0
# S_0为易感个体的初始比例
S_0 = 1 - I_0 - E_0 - R_0
# D_0为死亡个体初始比例
D_0 = 0
# T为传播时间
T = 180
# INI为初始状态下易感个体比例及感染个体比例
INI = (S_0, I_0, E_0, R_0, D_0)


def funcSI(prop, _):
    Y = np.zeros(5)
    X = prop
    # 易感个体变化
    Y[0] = - beta * X[0] * X[1]
    # 感染个体变化
    Y[1] = alpha * X[2] - gamma_2 * X[1] - die * X[1]
    # 潜伏期个体变化
    Y[2] = beta * X[0] * X[1] - (alpha + gamma_1) * X[2]
    # 治愈个体变化
    Y[3] = gamma_1 * X[2] + gamma_2 * X[1]
    # 死亡个体变化
    Y[4] = die * X[1]
    return Y


T_range = np.arange(0, T + 1)

RES = spi.odeint(funcSI, INI, T_range)
infection = RES[:, 1]

apex = np.where(infection == np.max(infection))[0][0]
safe_data = np.where(infection < 0.075)[0]
safe = 0
# print(safe_data)

for i in range(len(safe_data)):
    if safe_data[i + 1] - safe_data[i] > 10:
        safe = safe_data[i + 1]
        break
print(safe)
apex_date = datetime.datetime.strptime("2019-12-08", "%Y-%m-%d") + datetime.timedelta(days=int(apex))
safe_date = datetime.datetime.strptime("2019-12-08", "%Y-%m-%d") + datetime.timedelta(days=int(safe))

print(apex_date)
print(safe_date)
plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
plt.plot(RES[:, 1], color='red', label='Infection', marker='.')
plt.plot(RES[:, 2], color='orange', label='E', marker='.')
plt.plot(RES[:, 3], color='green', label='Recovery', marker='.')
plt.plot(RES[:, 4], color='black', label='Die', marker='.')
plt.title('SEIR-2019nCoV Model')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Proportion')
plt.show()
