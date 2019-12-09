password=t4AVmMv0ef0
i=0
for host in $(cat ../../cluster_scripts/nodes.txt); do
    sshpass -p "$password" ssh root@$host "cd elastic_rods/examples/actuation_sparsification && python3 node_task.py 15 $i" &
    i=$(($i + 1))
done
