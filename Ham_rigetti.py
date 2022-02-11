import numpy as np
from scipy import interpolate

def read_file(filename):
    x = []
    file_in = open(filename, 'r')
    for y in file_in.readlines():
       x.append(float(y.split()[0]))
    return x


def getLoweringOperators():

    # lowering oscillator for 3 levels, dimension 3x3:
    lower3 = np.matrix([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
    
    # Set up lowering operators for qubit 0 and 1 in full dimension 9x9:
    id3 = np.matrix(np.eye(3))
    a = []
    a.append( np.kron(lower3, id3) )
    a.append( np.kron(id3, lower3) )
    
    return a

def evalHd_mat():
    a = getLoweringOperators()

    # Qubit 0: Hd in full dimension 9x9:
    omega0 = 3.887 # in GHz
    xi0 = 0.187 # in GHz  
    Hd = omega0 * 2*np.pi * ( a[0].getH() * a[0] ) - xi0/2.0 * 2*np.pi * (a[0].getH() * a[0].getH() * a[0] * a[0]) 
    
    # Add coupling to Hd in full dimensions 9x9
    g01 = 0.01
    Hd += g01 * 2*np.pi * (a[0] + a[0].getH()) * (a[1] + a[1].getH())
    
    #print("Hd=\n", Hd)

    return Hd


## THIS IS A FUNCTION THAT QUANDARY REQUIRES to get the system Hamiltonian Hd! ##
# For now, the Hamiltonians here should be real valued!
# Return the vectorized Hamiltonian, column major vectorization (order='F')
# Quandary requires a *list* of elements. Therefore, need to cast to an array, than flatten it, then cast to a list.
def getHd():

    Hd = evalHd_mat()
    Hdlist = list(np.array(Hd).flatten(order='F'))
    return Hdlist


## THIS IS A FUNCTION THAT QUANDARY REQUIRES to get the control Hamiltonians Hc! ##
# For now, the Hamiltonians here should be real valued!
# Returns a list of lists of flattened Hamiltonians (also lists): 
# for each oscillator k=0,...Q-1:
#   for each control term i=0,...,C^k
#       Hc^k_i: control Hamiltonian stored as a flattened list.
def getHc():

    # Set up Hc^k_i 
    a = getLoweringOperators()
    # Qubit 0: no controls -> C^0 = 0 
    # Qubit 1: 2 controls -> C^1 = 2
    Hc10 = a[1].getH() * a[1]   # first conrol: Hc^1_0 = a^\dag a
    Hc11 = a[1].getH() * a[1].getH() * a[1] * a[1]   # first conrol: Hc^1_0 = a^\dag a^\dag a a

    # flatten Hc10 and Hc11 into lists
    Hc10list = list(np.array(Hc10).flatten(order='F'))
    Hc11list = list(np.array(Hc11).flatten(order='F'))

    # Set up return list
    Hclist = [ [], [Hc10list, Hc11list] ]  # Qubit 0: No control, Qubit 1: two controls Hc10 and Hc11
    #print("Hc10=",Hc10)
    #print("Hc11=",Hc11)

    return Hclist 


## THIS IS A FUNCTION THAT QUANDARY REQUIRES to get the transfer functions per control Hamiltonian! ##
# Should return a list of lists of functions, matching to the list of lists of flattened Hamiltonians in getHc
def getTransfer():

    # Transfer functions for oscillator 0: none
    # Transfer functions for oscillator 1: two function

    # read in x and y values to approximate omega and xi for Oscillator 1
    x     = read_file("./RigettiMatching/splinefit/x.dat")
    yfreq = read_file("./RigettiMatching/splinefit/omega.dat")
    yanha = read_file("./RigettiMatching/splinefit/xi.dat")
    
    # create splines
    order = 2
    tck_freq = interpolate.splrep(x, yfreq, s=0, k=order);
    tck_anha = interpolate.splrep(x, yanha, s=0, k=order);


    omega1 = [list(tck_freq[0]), list(tck_freq[1]), tck_freq[2]]
    xi1    = [list(tck_anha[0]), list(tck_anha[1]), tck_anha[2]]

    return [ [], [omega1, xi1] ]


def main():
    Hclist = getHc();
    #print(Hclist)

    transfer_func = getTransfer()
    print("Transfer functions for oscil0:", transfer_func[0])
    print("Transfer functions for oscil1:", transfer_func[1])


   
if __name__ == "__main__":
    main()
