import numpy

##########################################################
# Helper script to sort a control parameter file provided 
# by Juqbox into the format read by Quandary. 
####################################################

def main():

    # Set the file name of the original juqbox file (will be read in), and the output file name
    juqboxfile = "params_juqbox.dat"
    quandaryfile= juqboxfile + "_sorted_for_quandary"
    
    # Set the control parameter settings that had been used in  Juqbox
    Q = 2   # number of oscillators
    L = 10  # number of splines
    Nf = 1  # number of carrier wave frequencies (must here be the same for each oscillator!)
    
    
    ### Typically nothing to be done below here ###
    
    # Get juqbox data
    juqbox = readdata(juqboxfile)
    print("Number of designs: ", len(juqbox))
    
    # Map data from juqbox to quandary
    quandary=[]
    for k in range(Q):
        for l in range(L):
            for f in range(Nf):
                quandary.append(juqbox[k*2*L*Nf + f*2*L + l])
                quandary.append(juqbox[k*2*L*Nf + f*2*L + + L + l])
    
    
    # Print quandary params 
    writedata(quandaryfile, quandary)


# Helper funciton to read data from file (one column, returning as strings!)
def readdata(filename):
   print("Reading file: ", filename)
   # open file for read only
   ufile=open(filename,'r')
   data=[]
   # parse the lines of the file
   for i,line in enumerate(ufile):
     if (line.startswith('#')):     # ignore lines that begin with '#'
         continue
     line = line.split()
     #data.append(float(line[0]))
     data.append(line[0])
   #close the file
   ufile.close()
   return data

# Helper funciton to write data to file (data contains strings!)
def writedata(filename, data):
    print("Writing file: ", filename)
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()
