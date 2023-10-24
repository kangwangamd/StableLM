path = 'mpt-30b.stats.csv'

file1 = open(path, 'r')
Lines = file1.readlines()

def help_get_perc(line):
  tmp = line.strip()
  tmp = tmp.split(',')
  return float(tmp[-1])

total_time = 0.0
line_cnt = 0
RCCL, GEMM, elementwise, FA, barrier = 0.0, 0.0, 0.0, 0.0, 0.0
for line in Lines:
    line_cnt += 1
    if 'ccl' in line.strip():
      RCCL += help_get_perc(line)
    elif 'Cijk_Ailk' in line.strip():
      GEMM += help_get_perc(line)
    elif 'elementwise' in line.strip():
      elementwise += help_get_perc(line)
    elif 'grouped' in line.strip():
      FA += help_get_perc(line)
    elif 'barrier' in line.strip():
      barrier += help_get_perc(line)
    '''
    if line_cnt > 1:
        tmp = line.strip()
        tmp = tmp.split(',')
        total_time += float(tmp[2])
    '''

print(total_time / 1000000000.0)
print(elementwise)
print(RCCL)
print(GEMM)
print(FA)
other = 100 - elementwise - RCCL - GEMM - FA - barrier

items = [elementwise, RCCL, GEMM, FA, other]
new_base = elementwise + RCCL + GEMM + other

print()
for i in items:
    print(i / new_base)
