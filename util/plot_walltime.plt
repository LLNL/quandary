# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_walltime.plt

reset

set grid
set logscale x 2 
set logscale y 2 

set title "strong scaling"

p \
  datafile u 1:2 index 0 w lp t 'N=1000' ,\
  datafile u 1:2 index 1 w lp t 'N=2000' ,\
  datafile u 1:2 index 2 w lp t 'N=4000' ,\
  datafile u 1:2 index 3 w lp t 'N=8000' ,\
  datafile u 1:2 index 4 w lp t 'N=16000' ,\
  datafile u 1:2 index 5 w lp t 'N=32000' 


set term png
set output 'out.png'
replot
set term x11
replot

pause -1 "Plot 'out.png' written. Hit any key to continue"
