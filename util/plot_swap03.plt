reset

set grid
set xlabel 'time (ns)'
set ylabel 'expected energy level'
#set yrange [0:1]

#set term postscript dashed color
#set output 'out.ps'

set multiplot layout 2, 4
set title 'initial state 0001'
p \
    'expected0.iinit0017.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0017.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0017.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0017.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 0011'
p \
    'expected0.iinit0051.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0051.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0051.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0051.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 0101'
p \
    'expected0.iinit0085.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0085.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0085.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0085.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 0111'
p \
    'expected0.iinit0119.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0119.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0119.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0119.dat' u 1:2 w l t 'qubit 3', \


set title 'initial state 1000'
p \
    'expected0.iinit0136.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0136.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0136.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0136.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 1010'
p \
    'expected0.iinit0170.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0170.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0170.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0170.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 1100'
p \
    'expected0.iinit0204.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0204.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0204.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0204.dat' u 1:2 w l t 'qubit 3', \

set title 'initial state 1110'
p \
    'expected0.iinit0238.dat' u 1:2 w l t 'qubit 0', \
    'expected1.iinit0238.dat' u 1:2 w l t 'qubit 1', \
    'expected2.iinit0238.dat' u 1:2 w l t 'qubit 2', \
    'expected3.iinit0238.dat' u 1:2 w l t 'qubit 3', \


unset multiplot


#pause -1 "Plot 'out.ps' written. Hit any key to continue"
