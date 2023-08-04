path = 'run.log'

file1 = open(path, 'r')
Lines = file1.readlines()
for line in Lines:
    if 'FA_total_t' in line.strip():
        print(line.strip())

