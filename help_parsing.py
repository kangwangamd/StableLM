#!/usr/bin/env python3
import csv

path_non = 'stableML_non_flash_attn.stats.csv'
path_fa = 'stableML_flash_attn.stats.csv'
file_non = open(path_non, 'r')
file_fa = open(path_fa, 'r')


def get_key_dict(path):
    cnt = 0
    profile = dict()
    total_time = 0
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cnt += 1
            if cnt == 1:
                header = row
            else:
                key, calls, TotalDurationNs, AverageNs, Percentage = row
                total_time += int(TotalDurationNs)
                if key in profile:
                    print('sth is worng')
                profile[key] = (int(calls), int(TotalDurationNs))
    return profile, total_time


FA, FA_total_t = get_key_dict(path_fa)
non_FA, non_FA_total_t = get_key_dict(path_non)


# step 1 check total time diff?
print('check total time diff, FA, non-FA', FA_total_t, non_FA_total_t)
if FA_total_t > non_FA_total_t:
    print('Using Flash-attention is slower')
    diff_t = FA_total_t - non_FA_total_t
    print('time diff ==', diff_t)
    print('ratio', diff_t / non_FA_total_t)

# step 2 where is the diff coming from?

FA_only = list()
FA_same = list()
time_FA_only = 0
time_more = 0
for key, value in FA.items():
    calls, TotalDurationNs = value
    if key not in non_FA:
        FA_only.append((calls, key, TotalDurationNs))
        time_FA_only += TotalDurationNs
    else:
        if TotalDurationNs != non_FA[key][1]:
            FA_same.append((calls, key))
            if TotalDurationNs > non_FA[key][1]:
                time_more += (TotalDurationNs - non_FA[key][1])
            else:
                time_more -= (non_FA[key][1] - TotalDurationNs)

print('time_FA_more =', time_FA_only, time_more, time_FA_only + time_more)

print(len(FA_only), len(FA_same))
FA_only.sort(reverse=True)
FA_same.sort(reverse=True)

'''
for info in FA_only:
    print(info)
print()
for info in FA_same:
    print(info)
'''

print('========================================================')

non_FA_only = list()
non_FA_same = list()
time_non_FA_only = 0
for key, value in non_FA.items():
    if key not in FA:
        non_FA_only.append((value[0], key))
        time_non_FA_only += value[1]
    else:
        if non_FA[key][0] != FA[key][0]:
            non_FA_same.append((non_FA[key][0], key, FA[key][1]))
            if value[1] > FA[key][1]:
                time_non_FA_only += value[1] - FA[key][1]

print('time_non_FA_only =', time_non_FA_only)


print(len(non_FA_only), len(non_FA_same))
'''
non_FA_only.sort(reverse=True)
non_FA_same.sort(reverse=True)
for info in non_FA_only:
    print(info)

print()
for info in non_FA_same:
    print(info)
'''            


'''
cnt = 0
profile = dict()
key_set = set()
for line in file_non.readlines():
    cnt += 1
    if cnt == 1:
        header = line.split(',')
    else:
        if len(line.split(',')) == 5:
            kernel_func_key, calls, TotalDurationNs, AverageNs, Percentage = line.split(',')
            key = kernel_func_key[1:-1]
            #if profile[key]:
            #    print(key, ' already exist!')
            profile[key] = calls
        else:
            print(line.split(','))
print(len(profile), cnt)

    #if "MIOpenDriver" in line.strip():
    #    print(line.strip())
'''
