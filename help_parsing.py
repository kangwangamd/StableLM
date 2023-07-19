import csv

path_non = 'stableML_non_flash_attn.stats.csv'
path_fa = 'stableML_flash_attn.stats.csv'
file_non = open(path_non, 'r')
file_fa = open(path_fa, 'r')


def get_key_dict(path):
    cnt = 0
    profile = dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cnt += 1
            if cnt == 1:
                header = row
            else:
                key, calls, TotalDurationNs, AverageNs, Percentage = row
                if key in profile:
                    print('sth is worng')
                profile[key] = (int(calls), int(TotalDurationNs))
    return profile

FA = get_key_dict(path_fa)
non_FA = get_key_dict(path_non)


FA_only = list()
FA_same = list()
for key in FA:
    if key not in non_FA:
        FA_only.append((FA[key][0], key, FA[key][1]))
    else:
        if FA[key][0] != non_FA[key][0]:
            FA_same.append((FA[key][0], key))

print(len(FA_only), len(FA_same))
FA_only.sort(reverse=True)
FA_same.sort(reverse=True)

for info in FA_only:
    print(info)
print()
for info in FA_same:
    print(info)

print('========================================================')

non_FA_only = list()
non_FA_same = list()
for key in non_FA:
    if key not in FA:
        non_FA_only.append((non_FA[key][0], key))
    else:
        if non_FA[key][0] != FA[key][0]:
            non_FA_same.append((non_FA[key][0], key, FA[key][1]))

print(len(non_FA_only), len(non_FA_same))
non_FA_only.sort(reverse=True)
non_FA_same.sort(reverse=True)
for info in non_FA_only:
    print(info)

print()
for info in non_FA_same:
    print(info)
            


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
