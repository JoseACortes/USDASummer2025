
echo ########
# echo 'Running' detector_dfs_20_dfg_15
echo ########
echo ''
# input_file=test.inp

input_file=../Sims/7x7x7_Coconut_001022.txt

# input_file=template.txt

rm plotm.ps comout outp ptrac
mcnp6 r i=$input_file notek com=plotcom
mcnp6 p i=$input_file notek com=plotcom
echo ''
ps2pdf plotm.ps plott.pdf
rm comout outp runtpe mctal plotm.ps outq
date