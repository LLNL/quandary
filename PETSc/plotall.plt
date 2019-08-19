reset

set grid
set xlabel 'time'

p 'out_u.0000.dat' u 1:2 w l,\
  'out_u.0000.dat' u 1:3 w l,\
  'out_u.0000.dat' u 1:6 w l,\
  'out_u.0000.dat' u 1:7 w l,\
  'out_u.0000.dat' u 1:12 w l,\
  'out_u.0000.dat' u 1:13 w l,\
  'out_u.0000.dat' u 1:16 w l,\
  'out_u.0000.dat' u 1:17 w l,\
  'out_v.0000.dat' u 1:4  w l dt 2,\
  'out_v.0000.dat' u 1:5  w l dt 2,\
  'out_v.0000.dat' u 1:8  w l dt 2,\
  'out_v.0000.dat' u 1:9  w l dt 2,\
  'out_v.0000.dat' u 1:10 w l dt 2,\
  'out_v.0000.dat' u 1:11 w l dt 2,\
  'out_v.0000.dat' u 1:14 w l dt 2,\
  'out_v.0000.dat' u 1:15 w l dt 2,\

