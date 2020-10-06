#!/usr/bin/env python # # Script for testing if rho is hermetian, and trace(rho)=1. 
# rho = u + iv
# u and v are vectorized (columnwise)


import sys
from numpy import *
import math

def getdata(filename):

  # open file for read only
  ufile=open(filename,'r')

  data=[]

  print("Reading file ", filename, "...")
  # parse the lines of the file
  for i,line in enumerate(ufile):
    if (line.startswith('#')):     # ignore lines that begin with '#'
        continue
    linedata = parseline(line)
    data.append(linedata)
    #print(linedata)

  #close the file
  ufile.close()

  print("Done. ", len(data), " lines.")
  return data 

def parseline(line):
    string    = line.split()  # split the string in that line by separator " "
    floatlist = []
    for data in string:
        floatlist.append(float(data))  #parse the number in that line

    return floatlist 

# define error tolerance
EPS = 1E-10

# parse command line arguments
args = sys.argv
if (len(args) == 3):
  ufile  = str(args[1])  # File containing Re(rho) data
  vfile  = str(args[2])  # File containing Im(rho) data
elif (len(args) == 4):
  ufile  = str(args[1])  # File containing Re(rho) data
  vfile  = str(args[2])  # File containing Im(rho) data
  EPS    = float(args[3])       # Error tolerance
else:
  print("\nERROR while parsing argument list!\n")
  print("Usage: python3 sanitycheck.py <rho_Re_datafile> <rho_Im_datafile>")
  print("Usage: python3 sanitycheck.py <rho_Re_datafile> <rho_Im_datafile> <tolerance (default 1e-12)>\n")
  stop

# read data from reference and test file
udata = sorted(getdata(ufile))
vdata = sorted(getdata(vfile))

# Get matrix dimensions 
dim_vec = len(udata[0])
dim_mat = int(math.sqrt(dim_vec))
print("Density matrix dimension:  ", dim_mat, "x", dim_mat)
print("Using error tolerance: EPS =", EPS)


# Test 1: Check if rho is hermitian. 
print("\n Test 1: Is rho Hermitian?")
# Hence, u is symmetric, v is antisymmetric
isHermitian = True
hermitian_err_max = 0.0

#Iterate over the time steps
for ts in range(len(udata)):
    time  = udata[ts][0]
    timev = udata[ts][0]
    if (time != timev):
        print("ERROR: time in u and v don't match!")
        stop

    uline = udata[ts][1:]
    vline = vdata[ts][1:]
    #print("Testing time step", ts, ": t=", time, end=' ') 
    
    for i in range(dim_mat):
        for j in range(i,dim_mat):
            udiff = uline[i*dim_mat+j] - uline[j*dim_mat+i]
            vdiff = vline[i*dim_mat+j] + vline[j*dim_mat+i]
            if (abs(udiff) > EPS or abs(vdiff) > EPS):
                print("WARNING at t=", time, ": rho is not hermitian! Error: ", udiff, vdiff)
                isHermitian = False
                hermitian_err_max = max(hermitian_err_max, abs(udiff), abs(vdiff))

if (not isHermitian):
    print("   Test failed: Rho is NOT hermitian! Max error = ", hermitian_err_max)
else:
    print("   Success: Rho is hermitian!")
print()


# Test 2: Check if Trace(rho) = 1
# Hence, Trace(u) = 1 and Trace(V) = 0
print(" Test 2: Does rho have trace 1?")
hasTrace1 = True
trace1_err_max = 0.0

for ts in range(len(udata)):
    time  = udata[ts][0]
    timev = udata[ts][0]
    if (time != timev):
        print("ERROR: time in u and v don't match!")
        stop

    uline = udata[ts][1:]
    vline = vdata[ts][1:]
    #print("Testing time step", ts, ": t=", time,end=' ') 
 
    utrace = 0.0
    vtrace = 0.0
    for i in range(dim_mat):
       utrace += uline[i*dim_mat+i] 
       vtrace += vline[i*dim_mat+i]

    if (abs(vtrace - 0.0) > EPS):
         hasTrace1 = False
         trace1_err_max = max(trace1_err_max, abs(vtrace-0.0))
         print("WARNING at t=", time, ": Trace(v) is not 0.0! Tr(v)=", i, vtrace)
    if (abs(utrace - 1.0) > EPS):
         hasTrace1 = False
         trace1_err_max = max(trace1_err_max, abs(utrace-1.0))
         print("WARNING at t=", time, ": Trace(u) is not 1.0! Tr(u)=", utrace)

if (not hasTrace1):
    print("   Test failed: Rho does NOT have trace one! Max error = ", trace1_err_max) 
else:
    print("   Success: Rho has trace one! \n") 
print()


if (isHermitian and hasTrace1):
    print("\n SUCCESS! All tests passed!\n")
else:
    print("\n Some tests FAILED:\n")
    print("Is rho hermitian?    ", isHermitian)
    if not isHermitian:
        print("    Max err = ", hermitian_err_max)
    print("Trace of rho is 1.0? ", hasTrace1)
    if not hasTrace1:
        print("    Max err = ", trace1_err_max)
