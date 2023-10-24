path = 'mpt-13b/mpt-13b.stats.csv'

file1 = open(path, 'r')
Lines = file1.readlines()

def help_get_perc(line):
  tmp = line.strip()
  tmp = tmp.split(',')
  return float(tmp[-1])
  

RCCL, GEMM, elementwise, FA, barrier = 0.0, 0.0, 0.0, 0.0, 0.0
for line in Lines:
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

print('elementwise', elementwise)
print('RCCL', RCCL)
print('GEMM', GEMM)
print('FA', FA)
print('barrier', barrier)
print('else', 1- elementwise - RCCL - GEMM - FA - barrier)
