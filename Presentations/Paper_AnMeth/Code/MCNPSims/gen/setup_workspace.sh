compute_folder=compute/
mkdir $compute_folder
mkdir "$compute_folder"commands "$compute_folder"input "$compute_folder"output
mkdir "$compute_folder"output/outp "$compute_folder"output/runtpe "$compute_folder"output/mctal "$compute_folder"output/ptrac
echo "Simulation time will be recorded here." > "$compute_folder"output/simulation_time.txt

mkdir SimInfo
mkdir Spectrums