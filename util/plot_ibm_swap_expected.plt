reset

set grid
set xlabel 'time (ns)'
set ylabel 'expected energy level'
#set yrange [0:1]

set term postscript dashed color
set output 'out.ps'

set multiplot layout 2, 2
set title 'initial state 0x0'
p \
    'expected0.iinit0000.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0000.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 0x1'
p \
    'expected0.iinit0005.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0005.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 1x0'
p \
    'expected0.iinit0010.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0010.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 1x1'
p \
    'expected0.iinit0015.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0015.rank0000.dat' u 1:2 w l t 'Q2', \

unset multiplot


set term qt
set multiplot layout 2, 2
set title 'initial state 0x0'
p \
    'expected0.iinit0000.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0000.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 0x1'
p \
    'expected0.iinit0005.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0005.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 1x0'
p \
    'expected0.iinit0010.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0010.rank0000.dat' u 1:2 w l t 'Q2', \

set title 'initial state 1x1'
p \
    'expected0.iinit0015.rank0000.dat' u 1:2 w l t 'Q1', \
    'expected1.iinit0015.rank0000.dat' u 1:2 w l t 'Q2', \

unset multiplot


pause -1 "Plot 'out.ps' written. Hit any key to continue"
