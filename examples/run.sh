#!/bin/bash
example_dir=$(dirname "$0")
result_dir=$example_dir/../results
mkdir -p $result_dir
# ./open_linkage $example_dir/nonuniform_linkage.obj $example_dir/cross_sections/plus.msh 20 1.0 4
# mv open_it_0.msh $result_dir/nonuniform_closed.msh
# mv final_equilibrium.msh $result_dir/nonuniform_open.msh
# 
# ./open_linkage $example_dir/hyperboloid_open.obj $example_dir/cross_sections/plus.msh 16 -0.9 10
# mv open_it_0.msh $result_dir/hyperboloid_open_equilibrium.msh
# mv final_equilibrium.msh $result_dir/hyperboloid_closed_equilibrium.msh


# ./test_elastic_rod_linkage $example_dir/sphere_open.obj $example_dir/cross_sections/rectangle.msh 67
# mv equilibrium_config.msh $result_dir/sphere_open_equilibrium.msh

./open_linkage $example_dir/sphere_open.obj $example_dir/cross_sections/rectangle.msh 67 -0.4 10
mv open_it_0.msh sphere_open_equilibrium.msh
mv final_equilibrium.msh $result_dir/sphere_closed_equilibrium.msh
