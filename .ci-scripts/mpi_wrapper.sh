#!/bin/bash

mkdir -p /root/bin
MPI_PATH=$(command -v mpirun)
echo -e "#!/bin/bash\nexec $MPI_PATH --map-by :OVERSUBSCRIBE --allow-run-as-root \"\$@\"" > /root/bin/mpirun
chmod +x /root/bin/mpirun
echo 'export PATH="/root/bin:$PATH"' >> ~/.bashrc
