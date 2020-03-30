# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_optim.plt

# set a default file 
if (!exists("datafile")) datafile='optim.dat'

set grid
set logscale y
set format y "%g"
set y2tics
set ytics nomirror
set y2range[0:1.1]

set xlabel "Iterations"
set y2label "Fidelity"
set ylabel "Objective, gradient, inf du"

p \
    datafile u 2  w l t 'objective', \
    datafile u 4  w l t 'gradient', \
    datafile u 5  w l t 'inf du', \
    datafile u 3 axis x1y2 w l t 'fidelity'

pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot
#set term x11
#replot

