reset
set grid

set xlabel 'duration'
set ylabel 'expected energy level' 
set title 'Alice'

plot 'expected_alice.dat' using 1:2 with lines title 'p=q=1',\
     'expected_alice.dat' u 1:3 w l t 'p=q=2',\
     'expected_alice.dat' u 1:4 w l t 'p=q=3',\
     'expected_alice.dat' u 1:5 w l t 'p=q=4',\
     'expected_alice.dat' u 1:6 w l t 'p=q=5',\
     'expected_alice.dat' u 1:7 w l t 'p=q=6',\
     'expected_alice.dat' u 1:8 w l t 'p=q=7',\
     'expected_alice.dat' u 1:9 w l t 'p=q=8',\
     'expected_alice.dat' u 1:10 w l t 'p=q=9'
