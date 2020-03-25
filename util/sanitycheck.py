#!/usr/bin/env python # # Script for testing if rho is hermetian, and trace(rho)=1. 
# rho = u + iv
# u are vectorized columnwise. 


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

# parse command line arguments
args = sys.argv
if (len(args) == 3):
  ufile  = str(args[1])  # File containing u data
  vfile  = str(args[2])  # File containing v data
else:
  print("\nERROR while parsing argument list !")
  print("Usage: python3 sanitycheck.py <u_datafile> <v_datafile> \n")
  stop

# define some constants
EPS = 1E-14

# read data from reference and test file
udata = sorted(getdata(ufile))
vdata = sorted(getdata(vfile))

# Get matrix dimensions 
dim_vec = len(udata[0])
dim_mat = int(math.sqrt(dim_vec))
print("Density matrix dimension:  ", dim_mat, "x", dim_mat)


# Test 1: Check if rho is hermitian. 
# Hence, u is symmetric, v is antisymmetric
isHermitian = True

#Iterate over the time steps
for ts in range(len(udata)):
    time  = udata[ts][0]
    timev = udata[ts][0]
    if (time != timev):
        print("ERROR: time in u and v don't match!")
        stop

    uline = udata[ts][1:]
    vline = vdata[ts][1:]
    print("Testing time step", ts, ": t=", time, end=' ') 
    
    for i in range(dim_mat):
        for j in range(i,dim_mat):
            udiff = uline[i*dim_mat+j] - uline[j*dim_mat+i]
            vdiff = vline[i*dim_mat+j] + vline[j*dim_mat+i]
            if (abs(udiff) > EPS or abs(vdiff) > EPS):
                print("WARNING not hermitian! ",udiff, vdiff)
                isHermitian = False
            else:
                print(" ",i,j," ", end='')
    print()

if (not isHermitian):
    print("\n ERROR: Rho is NOT hermitian! \n")
else:
    print("\n Rho is hermitian! \n")


# Test 2: Check if Trace(rho) = 1
# Hence, Trace(u) = 1 and Trace(V) = 0
hasTrace1 = True

for ts in range(len(udata)):
    time  = udata[ts][0]
    timev = udata[ts][0]
    if (time != timev):
        print("ERROR: time in u and v don't match!")
        stop

    uline = udata[ts][1:]
    vline = vdata[ts][1:]
    print("Testing time step", ts, ": t=", time,end=' ') 
 
    utrace = 0.0
    vtrace = 0.0
    for i in range(dim_mat):
       utrace += uline[i*dim_mat+i] 
       vtrace += vline[i*dim_mat+i]
       if (abs(vtrace - 0.0) > EPS):
            print("WARNING: trace(v) is not 0.0!", i, vtrace)
            hasTrace1 = False
       else:
            print(" ",i, " ", end='')
    if (abs(utrace - 1.0) > EPS):
        hasTrace1 = False
        print("WARNING: trace(u) is not 1.0!", utrace)
    print()

if (not hasTrace1):
   print("\n ERROR: Rho does NOT have trace one! \n") 
else:
   print("\n Rho has trace one! \n") 


if (isHermitian and hasTrace1):
    print("\n SUCCESS! All tests passed!\n")
else:
    print("\n ERROR: Some tests failed:\n")
    print("Is rho hermitian?    ", isHermitian)
    print("Trace of rho is 1.0? ", hasTrace1)
