import numpy as np
from scipy import interpolate

# Utility function to set up the lowering operators a0 and a1 for a system of 2 qudits modelled with 3 levels
# returns [a0, a1] where a0 = lowering0 \kron I and  a1 = I \kron lowering1
def getLoweringOperators():

    # lowering oscillator for one qudit with 3 levels, dimension 3x3:
    lower3 = np.matrix([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
   
    # Identity matrix in qudit dimension 3x3
    id3 = np.matrix(np.eye(3))

    # Set up lowering operators for qubit 0 and 1 in the full dimensions 9x9:
    a = []
    a.append( np.kron(lower3, id3) )
    a.append( np.kron(id3, lower3) )
    
    return a

## This is a function that Quandary REQUIRES to get the (time-independent) system Hamiltonian ##
# Return the vectorized Hamiltonian, column major vectorization (order='F')
def getHd():

    # System parameters. (Here chosen arbitrarily for debugging, those numbers don't make much sense)
    omega0 = 4.0 *2*np.pi
    omega1 = 5.0 *2*np.pi
    rot0 = 3.9   *2*np.pi
    rot1 = 4.7   *2*np.pi
    xi0 = 0.2   *2*np.pi  
    xi1 = 0.3   *2*np.pi  
    g01 = 0.01  *2*np.pi

    # Set up the constant system Hamiltonian Hd in full dimension 9x9:
    a = getLoweringOperators()  # These are the lowering operators in the full dimensions 9x9
    Hd = \
        + (omega0-rot0) * ( a[0].getH() * a[0] ) - xi0/2.0 * (a[0].getH() * a[0].getH() * a[0] * a[0]) \
        + (omega1-rot1) * ( a[1].getH() * a[1] ) - xi1/2.0 * (a[1].getH() * a[1].getH() * a[1] * a[1])  \
        - g01 * a[0].getH() * a[0] * a[1].getH() * a[1]


    # Quandary expects a list of the vectorized hamiltonian.
    Hdlist = list(np.array(Hd).flatten(order='F'))
    return Hdlist



## These are two optional functions that Quandary uses to get the time-dependent coupling Hamiltonians : Re(Hd_l(t)), and Im(Hd_l(t))
# Returns a list of flattened Hamiltonians: 
# for each coupling term l=0,1,...:
#   Hdtl: time-dependent Hamiltonian, Hd_l(t), stored as a flattened list.
def getHdt_real():

    # real part of Jaynes-Cummings coupling for qudit 0 <-> 1 : (a0^dag a1 + a0 a1^dag)
    a = getLoweringOperators()
    Hdt = ( a[0].getH()*a[1] + a[0]*a[1].getH() )  

    # Quandary expects a list of vectorized Hamiltonians (list of lists of floats)
    Hdt_list = list(np.array(Hdt).flatten(order='F'))
    return [ Hdt_list ]

def getHdt_imag():

    # imaginary part of Jaynes-Cummings coupling for qudit 0 <-> 1 : (a0^dag a1 - a0 a1^dag)
    a = getLoweringOperators()
    Hdt = ( a[0].getH()*a[1] - a[0]*a[1].getH() )  

    # Quandary expects a list of vectorized Hamiltonians (list of lists of floats)
    Hdt_list = list(np.array(Hdt).flatten(order='F'))
    return [ Hdt_list ] 

## THese are optional functions that Quandary uses to get the transfer functions per time-dependent system Hamiltonian
# Returns a list of spline representations, matching the list of flattened time-dependent system Hamiltonians in getHdt_real/_imag
#   Each spline definition has the format [knots, coefficients, order]
def getHdtTransfer_real():

    # Transfer for the real part of Jaynes-Cummings coupling for qudit 0 <-> 1 : Jkl * cos(eta * t) (will be multiplying Re(Hd_l(t))
    Jkl = 0.01 * 2*np.pi
    rot0 = 3.9 * 2*np.pi
    rot1 = 4.7 * 2*np.pi
    eta = rot0 - rot1

    # Transfer functions for Jkl*cos(eta*t)
    x = np.arange(0,300, 0.01)   # make sure this range is wide enough to cover eta*t for all time points t!
    y = Jkl * np.cos(eta*x)

    # create splines from scipy's interpolate module
    order = 2
    tck_cos = interpolate.splrep(x, y, s=0, k=order);    # returns [knots, coeffs, order], but have to make them lists

    y_spline = [list(tck_cos[0]), list(tck_cos[1]), tck_cos[2]]
    return [ y_spline ]

def getHdtTransfer_imag():

    # Transfer for the imaginary part of Jaynes-Cummings coupling for qudit 0 <-> 1 : Jkl * sin(eta * t) (will be multiplying Im(Hd_l(t))
    Jkl = 0.01 * 2*np.pi
    rot0 = 3.9 * 2*np.pi
    rot1 = 4.7 * 2*np.pi
    eta = rot0 - rot1

    # Transfer functions for the identity
    x = np.arange(0,300, 0.01) # make sure this range is wide enough to cover eta*t for all time points t!
    y = Jkl * np.sin(eta*x)

    # create splines from scipy's interpolate module
    order = 2
    tck_freq = interpolate.splrep(x, y, s=0, k=order);   # return [knots, coeffs, order], but have to make them lists

    y_spline = [list(tck_freq[0]), list(tck_freq[1]), tck_freq[2]]
    return [ y_spline ]




## These are functions that Quandary requires to get the control Hamiltonians Re(Hc^{k,l}(t)), Im(Hc^{k,l}), for each oscilator k
# Returns a list of lists of flattened Hamiltonians:
# for each oscillator k=0,...Q-1:
#   for each control term i=0,...,C^k
#       Hc^{k,l}: control Hamiltonian stored as a flattened list.
def getHc_real():

    a = getLoweringOperators()

    # Here, one control term per qudit: (a0 + a0^dag) for qudit 0 and (a1 + a1^dag) for qudit 1
    Hc0 = ( a[0] + a[0].getH() )  
    Hc1 = ( a[1] + a[1].getH() )  

    #  Prepare the return list of lists of Hamiltonians
    Hc0_list = list(np.array(Hc0).flatten(order='F'))
    Hc1_list = list(np.array(Hc1).flatten(order='F'))
    return [ [ Hc0_list ], [ Hc1_list ] ]

def getHc_imag():

    a = getLoweringOperators()

    # Here, one control term per qudit: (a0 - a0^dag) for qudit 0 and (a1 - a1^dag) for qudit 1
    Hc0 = ( a[0] - a[0].getH())  
    Hc1 = ( a[1] - a[1].getH())  

    #  Prepare the return list of lists of Hamiltonians
    Hc0_list = list(np.array(Hc0).flatten(order='F'))
    Hc1_list = list(np.array(Hc1).flatten(order='F'))
    return [ [ Hc0_list ], [ Hc1_list ] ] 


## These are optional functions that Quandary uses to get transfer functions u^k,l(p) applied to each control Hamiltonian: u^{k,l}(p^k(t)) * Hc^{k,l}
# Return a list of lists of spline representations (matching the list of lists of flattened Hamiltonians in getHc)
#   Each spline representation has the format [knots, coefficients, order]
def getHcTransfer_real():

    # Here, we don't have transfer functions on the control terms (we apply p(t) directly to Hc). However, I'm setting them up here as an example use, using the identity as a transfer function. 
    x = np.arange(-10,10) # Make sure this range is big enough to cover all possible input values p(t)
    y = x   # Here, the identity

    # create splines using scipy's interpolate module
    order = 2
    tck_freq = interpolate.splrep(x, y, s=0, k=order);    # returns [knots, coeffs, order], but need to make them lists. 

    iden = [list(tck_freq[0]), list(tck_freq[1]), tck_freq[2]]
    return [ [ iden ], [ iden ] ]  # See how this matches to the return list of lists in getHc_real()

def getHcTransfer_imag():

    # Here, we don't have transfer functions on the control terms (we apply q(t) directly to Hc. However, I'm setting them up here as an example use. 
    x = np.arange(-10,10)
    y = x

    # create splines using scipy's interpolate module
    order = 2
    tck_freq = interpolate.splrep(x, y, s=0, k=order);

    iden = [list(tck_freq[0]), list(tck_freq[1]), tck_freq[2]]
    return [ [ iden ], [ iden ] ]  # See how this matches to the return list of lists in getHc_imag



def main():
    print("Hello. Use main to check your Hamiltonian terms and matrices if you want...")

    print("Constant system matrix Hd:")
    Hd = np.reshape(getHd(), (9,9), order='F')   # Reshape from flattened list to matrix (column major)
    print(Hd)
       
if __name__ == "__main__":
    main()
