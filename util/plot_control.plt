# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_control.plt

reset
set grid

set title datafile 
set xlabel "time"
set ylabel "rotating frame controls"

p \
    datafile u 1:2  w l lc 3 t 'p(t)' ,\
    datafile u 1:3  w l lc 7 t 'q(t)' ,\


set term postscript dashed color
set output 'out.ps'
replot
set term qt
replot


pause -1 "Plot 'out.ps' written. Hit any key to continue"
