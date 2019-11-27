reset

# run from command line with 
# gnuplot -e "udata='udatafile.dat'" -e "vdata='vdatafile.dat'" plot_allcolumns.plt

# set default datafiles
if (!exists("udata")) udata='out_u.0000.dat'
if (!exists("vdata")) vdata='out_v.0000.dat'

# get number of columns from awk on udata file
syscommand = "awk 'NR==1{print NF}' ".udata
ncols = system(syscommand)

# set styles 
set xlabel 'time'
set grid

# Plot u
set title "u data"
plot \
    for [i=2:ncols] udata u 1:i w l t "Column ".i

pause -1 "Hit any key to continue"

# Plot v
set title "v data"
plot \
    for [i=2:ncols] vdata u 1:i w l t "Column ".i


pause -1 "Hit any key to continue"
