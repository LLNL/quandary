# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_diag.plt

reset

set key outside
set key horizontal 
set grid

set title datafile 
set xlabel "Time"
set ylabel "Population"

p \
    datafile u 1:2  w l lc 3 t '00' ,\
    datafile u 1:7  w l lc 7 t '01' ,\
    datafile u 1:12 w l lc 2 t '10' ,\
    datafile u 1:17 w l lc 1 t '11' 

pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot

