module purge
module load Anaconda3/2020.07

sacct -X -u $1 -S 2022-11-1 --format="Elapsed, Partition" > tmp.txt

python -c "
from datetime import timedelta

with open('tmp.txt', 'r') as f:
    l = f.readlines()

times = l[2:]
total_time = timedelta()
for t in times:
    elapsed, partition = t.split()
    if partition.strip() == 'GPUQ':
        h, m, s = [int(x) for x in elapsed.strip().split(':')]
        total_time += timedelta(hours=h, minutes=m, seconds=s)

print(total_time)
"

rm tmp.txt

