import numpy as np


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
    
    # Qubit1: omega_1(x)
    def omega1(flux):
        out = 0.444*np.sin(2.0*np.pi*(flux+0.25))+4.826 # sin fit
        # print("Inside omega1(", flux, ")");
        return out

    # Qubit1: xi_1(x)
    def xi1(flux):
        out = 0.00194*np.sin(2.0*np.pi*(flux+0.25)) - 0.19904
        # print("Inside xi1(", flux, ")");
        return out

    return [ [], [omega1, xi1] ]


def main():
    Hclist = getHc();
    #print(Hclist)


    #Hd = evalHd_mat()

    #with open('Bd_test.txt', 'r') as f:
    #    l = [[float(num) for num in line.split()] for line in f]
    ##print(l)
    #Bd_test = np.matrix(l)
    #print("Bd_test =", Bd_test)

    #for i in range(9):
    #    for j in range(9):
    #        Hd[i,j] = round(Hd[i,j], 4)
    #print("Hd=", Hd)

    ## test -In \kron Hd + Hd \kron In
    ##idn = np.eye(9)
    ##Hd = - np.kron(idn, Hd) + np.kron(Hd, idn)
    #
    #err = np.linalg.norm(Hd- Bd_test)
    #print("ERROR = ",err);
    #print("Diff = ",Hd-Bd_test)
    
if __name__ == "__main__":
    main()
