import sys

def compare_two_files(basefile, currentfile):

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

    #compute relative difference for the first column
    for j in range(ncol):
        for i in range(nrow):
            if abs(base_values[i][j]) > 10e-15:
                error = abs(base_values[i][j] - current_values[i][j])/abs(base_values[i][j])
            else:
                error = abs(base_values[i][j] - current_values[i][j])
       
            if error > 1.0e-3:
                return sys.exit(1)
    return sys.exit(0)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    compare_two_files(*sys.argv[1:])
