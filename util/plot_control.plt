# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_diag.plt

reset
set grid

set title datafile 
set xlabel "Time"
set ylabel "Control"

p \
    datafile u 2:3  w l lc 3 t 'Real' ,\
    datafile u 2:4  w l lc 7 t 'Imaginary' ,\

set term png
set output 'out.png'
replot
set term x11
replot

pause -1 "Plot 'out.png' written. Hit any key to continue"
