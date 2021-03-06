reset

# run from command line with 
# gnuplot -e "datafile='datafile.dat'" plot_fullstate.plt

# get number of columns from awk on datafile file
syscommand = "awk 'NR==1{print NF}' ".datafile
ncols = system(syscommand)

# set styles 
set xlabel 'time'
set grid

# Plot
plot \
    for [i=2:ncols] datafile u 1:i w l t "Column ".i

pause -1 "Plot 'out.png' written. Hit any key to continue"

set term png
set output 'out.png'
replot
#set term x11
#replot

