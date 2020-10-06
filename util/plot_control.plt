# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_diag.plt

reset
set grid

set title datafile 
set xlabel "time"
set ylabel "rotating frame controls"

p \
    datafile u 1:2  w l lc 3 t 'p(t)' ,\
    datafile u 1:3  w l lc 7 t 'q(t)' ,\

pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot
#set term x11
#replot

