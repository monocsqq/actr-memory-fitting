from __future__ import annotations
import actr
import concurrent.futures as cf
import csv
import cv2
from decimal import Decimal
from functools import lru_cache
import Levenshtein as lev
from multiprocessing import Manager
import numpy as np
import os
from pathlib import Path
import pandas as pd
import string
import sys
import random

ins_list = {}
port_base = 3101
for _ in range(100):
    port_num = _ + port_base
    ins_list[port_num] = None

df = None

# Run 5 times and take the average
# Launch with run_100_actr.bat
# target = '20240303_listonly_normalized'

# Launch with run_100_actr_ori_grouped.bat
target = '20241225_screened' # expt2 analysis1

trial = str(sys.argv[1])

outputs_dir = f'outputs{trial}'
hists_dir = f'hists{trial}'
nhists_dir = f'nhists{trial}'



# Temporary list for storing responses
response = {}
for _ in range(100):
    response[_ + port_base] = []

human_hist = []

# List of parameters to search
mp_list = []
rt_list = []

# Number of simulations to run per parameter combination
retrieve_num = 100

# Variables to store results
max_hi_sum = 0.0
best_mp_median = np.inf
best_mp_range = np.inf
best_rt_median = np.inf
best_rt_range = np.inf

# For time measurement
run_time = 0.0
make_model_hist_time = 0.0
hist_div_sum_time = 0.0
calc_hi_sum_time = 0.0
simulate_cached_time = 0.0
simulate_time = 0.0

import uuid

def open_unique_text(path: str | Path, mode: str = "w", encoding: str = "utf-8"):
    """
    If the path already exists, create a file by appending a _<uuid8> suffix.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok = True)

    stem = base.stem
    suffix = base.suffix
    parent = base.parent

    try:
        return (parent / f"{stem}{suffix}").open(mode="x", encoding=encoding)
    except FileExistsError:
        pass

    while True:
        tag = uuid.uuid4().hex[:8]
        cand = parent / f"{base.stem}_{tag}{suffix}"
        try:
            return cand.open(mode="x", encoding=encoding)
        except FileExistsError:
            continue


def print_human_hist ():

    print(human_hist)

def record_response (item):
    port_num = item[0]
    value = item[1]
    global response # Use a dict keyed by port number
    response[port_num].append(value)

def recall (mp, rt, port_num):

    ins_list[port_num].add_command("grouped-response",record_response,"Response recording function for the tutorial grouped model.")
    global response
    response[port_num] = []
    actr.reset(ins_list[port_num])
    # 100 combinations: mp(10) x rt(10)
    actr.set_parameter_value(":mp", mp, ins_list[port_num])
    actr.set_parameter_value(":rt", rt, ins_list[port_num])
    actr.run(20, connection=ins_list[port_num])
    actr.remove_command("grouped-response", ins_list[port_num])
    return response[port_num]

def run (mp, rt, port_num):

    output = Path(f"./python/{outputs_dir}/mp" + str(mp) + "_rt" + str(rt) + "_output.txt")
    if not os.path.exists(output):
        with open(output, 'w') as f:
            for _ in range(retrieve_num):
                recall(mp, rt, port_num)
                f.writelines(response[port_num])
                f.write('\n')
            f.close()

def make_model_hist (mp, rt):

    output = Path(f"./python/{outputs_dir}/mp{mp}_rt{rt}_output.txt")
    hist = Path(f"./python/{hists_dir}/mp{mp}_rt{rt}_hist.txt")
    model_hist = np.zeros(100, dtype=int)
    if os.path.exists(output):
        with open(output, 'r') as o:
            # que = "1234567890" # Use this for Experiment 1 data
            que = '123456789' # Experiment 2 data
            for _ in range(retrieve_num):
                line = o.readline()
                line = line.replace('\n', '')
  
                ans = str(line)
                dist = lev.distance(que, ans)
                model_hist[dist] += 1
            o.close()
    if not os.path.exists(hist):
        np.savetxt(hist, model_hist)

def hist_div_sum (mp, rt):

    histfile = Path(f"./python/{hists_dir}/mp{mp}_rt{rt}_hist.txt")
    if os.path.exists(histfile):
        hist = np.loadtxt(histfile)
    else:
        return -1
    nhist = Path(f"./python/{nhists_dir}/mp{mp}_rt{rt}_nhist.txt")
    if not os.path.exists(nhist):
        div = hist.sum()
        hist = np.where(div != 0, hist / div, 0.0)
        hist = hist[0:11]
        np.savetxt(nhist, hist)
    
def initialize_dataframe (human_hist_path):

    human_hist_list = []
    with open(human_hist_path, 'r') as human_data:
        for line in human_data:
            human_hist = np.array(eval(line), dtype=np.float32)
            human_hist_list.append(human_hist.tolist())
    
    human_hist_len = len(human_hist_list)

    # Create the initial DataFrame
    df = pd.DataFrame({
        "human_hist": human_hist_list,
        "hi": [np.nan] * human_hist_len,
        "corr": [np.nan] * human_hist_len,
        "mp": [np.nan] * human_hist_len,
        "rt": [np.nan] * human_hist_len,
    })
    print(df)
    return df


def calc_hi_sum (mp, rt):

    global df

    model_hist_path = Path(f"./python/{nhists_dir}/mp{mp}_rt{rt}_nhist.txt")
    hi_sum = 0.0
    model_hist = np.loadtxt(model_hist_path)
    model_hist = model_hist.astype(np.float32)

    # Compute hi and update the DataFrame
    for index, row in df.iterrows():
        human_hist = np.array(row["human_hist"], dtype=np.float32)
        hi = cv2.compareHist(human_hist, model_hist, cv2.HISTCMP_INTERSECT)

        # Update if hi is NaN or the new hi is larger
        if pd.isna(row["hi"]) or hi > row["hi"]:
            df.at[index, "hi"] = hi
            corr = cv2.compareHist(human_hist, model_hist, cv2.HISTCMP_CORREL)
            df.at[index, "corr"] = corr
            df.at[index, "mp"] = mp
            df.at[index, "rt"] = rt

        hi_sum += hi
    return hi_sum

@lru_cache(maxsize=100)
def simulate_cached (mp, rt, shared_list, count):
    global simulate_cached_time
    run(mp, rt, count)
    make_model_hist(mp, rt)
    hist_div_sum(mp, rt)
    hi_sum = calc_hi_sum(mp, rt)
    shared_list.append(hi_sum)
    return hi_sum

# Run all mp*rt (100 combinations) in parallel
def simulate ():

    global mp_list, rt_list, simulate_time
    hi_sum = 0.0
    for mp in mp_list:
        for rt in rt_list:
                hi_sum += simulate_cached(mp, rt)
    
    return hi_sum

def simulate_parallel():
    global mp_list, rt_list
    hi_sum = 0.0

    shared_list = Manager().list()

    with cf.ThreadPoolExecutor() as executor:
        # Run simulations in parallel for different mp/rt combinations
        count = 3101
        futures = []
        for mp in mp_list:
            for rt in rt_list:
                future = executor.submit(simulate_cached, mp, rt, shared_list, count)
                futures.append(future)
                count += 1

        # Collect simulation results
        for future in cf.as_completed(futures):
            hi_sum += future.result()

    return hi_sum
            

def create_list (median, range_v):

    low = median - range_v / 2
    high = median + range_v / 2
    size = 10
    array = np.linspace(low, high, size).tolist()
    return array

# If the result file (result_*.txt/csv) already exists, append a random string before saving
def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def main ():

    if not os.path.exists(Path(f'./python/outputs{trial}')):
        print(f'"./python/outputs{trial}" doesn\'t exists')
        os.makedirs(Path(f'./python/outputs{trial}'))
    else:
        print(f'"./python/outputs{trial}" exists')
    
    if not os.path.exists(Path(f'./python/hists{trial}')):
        print(f'"./python/hists{trial}" doesn\'t exists')
        os.makedirs(Path(f'./python/hists{trial}'))
    else:
        print(f'"./python/hists{trial}" exists')
    
    if not os.path.exists(Path(f'./python/nhists{trial}')):
        print(f'"./python/nhists{trial}" doesn\'t exists')
        os.makedirs(Path(f'./python/nhists{trial}'))
    else:
        print(f'"./python/outputs{trial}" exists')

    for _ in range(100):
        port_num = _ + port_base
        print(port_num)
        ins = actr.connection(port=port_num, connection=ins_list[port_num])
        ins.add_command("Python-import-from-file",actr.env_loader,"Import a Python module and make it available directly from the interactive prompt. Params: pathname")  
        ins.add_command("load-python-module-html",actr.env_loader_no_path,"Import a python module from the directory containing the actr.py module and make it available directly from the interactive prompt. Params: filename")
        ins_list[port_num] = ins
    
    global max_hi_sum, mp_list, rt_list, df

    human_hist_path = Path(f"./python/data{target}.txt") # exp 20241225

    df = initialize_dataframe(human_hist_path)
    max_hi_sum = -1.0
    med_increment = ["-0.5", "0.0", "0.5"]
    ran_increment = ["0.8", "1.0", "1.2"]
    temp_hi_sum = 0.0
    current_mp_median = 1.0
    current_mp_range = 1.0
    current_rt_median = 1.0
    current_rt_range = 1.0
    best_mp_median = np.inf
    best_mp_range = np.inf
    best_rt_median = np.inf
    best_rt_range = np.inf
    loop = 0
    while True:
        temp_hi_sum_max = -1.0
        for mp_median_inc in med_increment:
            for mp_range_inc in ran_increment:
                for rt_median_inc in med_increment:
                    for rt_range_inc in ran_increment:
                        temp_mp_median = float(Decimal(current_mp_median) + Decimal(mp_median_inc))
                        temp_mp_range = float(Decimal(current_mp_range) * Decimal(mp_range_inc))
                        temp_rt_median = float(Decimal(current_rt_median) + Decimal(rt_median_inc))
                        temp_rt_range = float(Decimal(current_rt_range) * Decimal(rt_range_inc))
                        print(f"mp中央値: {temp_mp_median}, mp範囲: {temp_mp_range}, rt中央値: {temp_rt_median}, rt範囲: {temp_rt_range}")
                        mp_list = create_list(temp_mp_median, temp_mp_range)
                        rt_list = create_list(temp_rt_median, temp_rt_range)
                        temp_hi_sum = simulate_parallel()
                        if temp_hi_sum > temp_hi_sum_max:
                            print(f"途中経過: {temp_hi_sum}")
                            temp_mp_median_max = temp_mp_median
                            temp_mp_range_max = temp_mp_range
                            temp_rt_median_max = temp_rt_median
                            temp_rt_range_max = temp_rt_range
                            temp_hi_sum_max = temp_hi_sum
        if temp_hi_sum_max > max_hi_sum:
            loop += 1
            current_mp_median = temp_mp_median_max
            current_mp_range = temp_mp_range_max
            current_rt_median = temp_rt_median_max
            current_rt_range = temp_rt_range_max
            max_hi_sum = temp_hi_sum_max
            print("######################################")
            print(f"HI最大値の更新: {max_hi_sum}")
            print(f"mp中央値: {current_mp_median}")
            print(f"mp範囲: {current_mp_range}")
            print(f"rt中央値: {current_rt_median}")
            print(f"rt範囲: {current_rt_range}")
            print("loop: " + str(loop))
            print("######################################")
            print(df.head())
            print("######################################")
        else:
            best_mp_median = current_mp_median
            best_mp_range = current_mp_range
            best_rt_median = current_rt_median
            best_rt_range = current_rt_range
            break
    
    print("################# result ################")
    print(f"HI最大値: {max_hi_sum}")
    print(f"mp中央値: {best_mp_median}")
    print(f"mp範囲: {best_mp_range}")
    print(f"rt中央値: {best_rt_median}")
    print(f"rt範囲: {best_rt_range}")
    print("loop: " + str(loop))
    print("#########################################")
    with open_unique_text(f"result{target}_{trial}.txt", 'w') as f:
        f.write(f"max_hi_sum : {max_hi_sum}\n")
        f.write(f"best_mp_median : {best_mp_median}\n")
        f.write(f"best_mp_range : {best_mp_range}\n")
        f.write(f"best_rt_median : {best_rt_median}\n")
        f.write(f"best_rt_range : {best_rt_range}\n")
        f.write("loop: " + str(loop) + "\n")
        f.close()
    simulate_cached.cache_clear()

    out_csv = f'./fit_{target}_{trial}.csv'
    if os.path.exists(out_csv):
        print('outfile already exists')
        rand_str = generate_random_string()
        out_csv = f'./fit_{target}_{trial}_{rand_str}.csv'
    df.to_csv(out_csv)
    print(f'saved to {out_csv}')

if __name__=="__main__":
    main()