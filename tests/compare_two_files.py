import sys
import numpy as np

def compare_two_files(basefile, currentfile, tolerance, isBitWiseStr):

    def extract_data(filename):                                                        
        infile = open(filename, 'r')                                                   
        numarray = []
        nrow = 0
        n = 0
        for line in infile:         
            if not line.startswith('#'):
                nrow = nrow + 1
                words = line.split()   
                n = len(words)
                index = 0;
                numbers = [] 
                for word in words:
                    number = float(word)                                                   
                    numbers.append(number)   
                numarray.append(numbers)
        infile.close()                                                                 
        return nrow, n, numarray                                                                 

    nrow, ncol, base_values = extract_data(basefile)                                    
    nrow2, ncol2, current_values = extract_data(currentfile)
    isBitWise = int(isBitWiseStr)

    if isBitWise:
        #compute relative difference bitwise 
        overall_error = 0.0
        for j in range(ncol):
            for i in range(nrow):
                error = abs(base_values[i][j] - current_values[i][j])
                if error > 0.0:
                    print("-- Error is too big ", error)
                    return sys.exit(1)
                else:
                    overall_error = np.maximum(overall_error, error)

        print("-- Test passed! error is ", overall_error)
        return sys.exit(0)
    else:
        #compute relative difference as whole 
        difference = 0.0
        base = 0.0
        overall_error = 0.0
        for j in range(ncol):
            for i in range(nrow):
                base = base + base_values[i][j]*base_values[i][j]
                diff_comp = base_values[i][j] - current_values[i][j]
                difference = difference + diff_comp*diff_comp
            sqrt_base = np.sqrt(base)
            sqrt_difference = np.sqrt(difference)
            error = 0.0
            if abs(sqrt_base) > 1e-15:
                error = sqrt_difference/sqrt_base #relative error
            else:
                error = sqrt_difference

            difference = 0.0
            base = 0.0

            if error > float(tolerance):
                print("-- Error is too big ", error)
                assert False
            else:
                overall_error = np.maximum(overall_error, error)
        print("-- Test passed! error is ", overall_error)
        assert True


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    compare_two_files(*sys.argv[1:])
