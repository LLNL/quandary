# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_optim.plt

# set a default file 
if (!exists("datafile")) datafile='optim.dat'

set grid
set logscale y
set format y "%g"

set xlabel "Iterations"

p \
    datafile u 2  w l t 'objective', \
    datafile u 3  w l t 'gradient', \
    datafile u 4  w l t 'inf du' 

set term png
set output 'out.png'
replot
set term x11
replot

pause -1 "Plot 'out.png' written. Hit any key to continue"
