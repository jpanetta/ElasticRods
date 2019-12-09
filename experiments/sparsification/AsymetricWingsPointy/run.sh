eta=0.1
p=0.5

if [ $# -eq 2 ]; then
    eta=$1
    p=$2
fi;

mkdir -p results

suffix=${eta}_L${p}_$trial
optimized_data=../../../python/examples/optimized/data/AsymmWingsPointy
actuation_sparsifier $optimized_data/deployed_opt.msh ../../../examples/cross_sections/Rectangle_12x8.json 2.0 $optimized_data/design_parameters.txt $eta $p > stdout.txt 2>stderr.txt
