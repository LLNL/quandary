# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_optim.plt

# set a default file 
if (!exists("datafile")) datafile='optim_history.dat'

#reset
#datafile='data_out/optim_history.dat'

set grid
set logscale y
set format y '$10^{%T}$'
set y2tics
set ytics nomirror
set y2range[0:1.05]
#set yrange [1e-7:1]
#set xrange [0:85]

#set key at 190, 40.0
set key bottom left


set xlabel 'iteration'
set y2label 'Fidelity'
set ylabel '$J, \|\nabla J\|$'

plot \
    datafile u 2 axis x1y1  w l lw 3 t '$J$', \
    datafile u 3 axis x1y1  w l lw 3 t '$\|\nabla J\|$', \
    datafile u 5 axis x1y2  w l lw 3 t 'Fidelity', \
    datafile u 6 axis x1y1  w l lw 3 t 'Cost', \
    datafile u 7 axis x1y1  w l lw 3 t '$\gamma_2||\alpha||$', \
#    datafile u 8 axis x1y1  w l t 'penalty', \


set term epslatex color size 15.5cm, 8cm
set output 'optim_history.tex'
replot


set term epslatex color size 12.5cm, 8cm standalone 
set output 'optim_history_standalone.tex'
replot

set term qt
replot

#set term postscript dashed color
#set output 'out.ps'
#replot
#pause -1 "Plot 'out.ps' written. Hit any key to continue"
#set term x11
#replot

