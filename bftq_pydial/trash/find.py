

f = open("tmp/param_search.cfg/results")
g = open("tmp/param_search.cfg/params")


lines = f.readlines()
lines_params = g.readlines()

for i,line in enumerate(lines):
    split = line.split()
    R = split[-2].split('[')[-1].split(']')[0].split(';')
    C = split[-1].split('[')[-1].split(']')[0].split(';')
    if float(R[1])> 7:
        print "{}, R=[{};{}] C=[{},{}] {}".format(i,float(R[0]),float(R[1]), float(C[0]),float(C[1]),lines_params[i])
