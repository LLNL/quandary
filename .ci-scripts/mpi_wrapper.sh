#!/bin/bash

mkdir -p /root/bin
MPI_PATH=$(command -v mpirun)
cat > /root/bin/mpirun << EOF
#!/bin/bash
exec $MPI_PATH --map-by :OVERSUBSCRIBE --allow-run-as-root "\$@"
EOF
chmod +x /root/bin/mpirun
echo 'export PATH="/root/bin:$PATH"' >> ~/.bashrc
