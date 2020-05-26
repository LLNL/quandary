# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_diag.plt
#
# set a default file 
if (!exists("datafile")) datafile='expected.iinit0000.rank0000.dat'

set xlabel 'time (us)'
set ylabel 'expected'
set grid
set xtics 0.2

p 'expected0.iinit0000.rank0000.dat'  u 1:2 w l t '00', \
  'expected0.iinit0004.rank0000.dat'  u 1:2 w l t '11', \
  'expected0.iinit0008.rank0000.dat'  u 1:2 w l t '22', \


pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot
#set term x11
#replot

