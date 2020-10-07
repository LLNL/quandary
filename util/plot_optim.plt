# run from command line with
# > gnuplot -e "datafile='data.dat'" plot_optim.plt

# set a default file 
if (!exists("datafile")) datafile='optim.dat'

set grid
set logscale y
set format y "%g"
#set y2tics
#set ytics nomirror
#set y2range[0:1.1]

set xlabel "iteration"
#set y2label "objective"
#set ylabel "||Pr(gradient)||"

p \
    datafile u 2 axis x1y2 w l t 'objective', \
    datafile u 3 axis x1y1  w l t '||Pr(gradient)||', \
    datafile u 5 axis x1y1  w l t 'terminal cost', \
    datafile u 6 axis x1y1  w l t 'tikhonov', \
    datafile u 7 axis x1y1  w l t 'penalty', \

pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot
#set term x11
#replot

