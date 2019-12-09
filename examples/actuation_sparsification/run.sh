eta=0.02
p=0.5
trial=0

if [ $# -eq 3 ]; then
    eta=$1
    p=$2
    trial=$3
fi;

mkdir -p results

suffix=${eta}_L${p}_$trial
OMP_NUM_THREADS=6 actuation_sparsifier half_vase_opened.msh ../cross_sections/PlusBeam_2x2.json 1.160016671967439 half_vase_rest_lengths.txt $eta $p > results/stdout_$suffix.txt
scp results/stdout_$suffix.txt jpanetta@mac220bc73d76.dyn.epfl.ch:cluster_results/
mv torques.txt results/torques_$suffix.txt
mv post_sparsification.msh results/post_sparsification_$suffix.msh
