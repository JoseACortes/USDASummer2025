# %%
import pandas as pd
import os

filenames = '../../sims_05.csv'
filenames = pd.read_csv(filenames)

compute_folder = '../../compute/'
commands_folder = compute_folder+'commands/'

# %%
sbatch = []
batch = []
with open('command_template.sh', 'r') as file:
    template = file.read()
    file.close()
    
for name in filenames['filename']:
    name = name.split('.')[0]

    # replace "@filename@" with the filenames
    temp = template
    temp = temp.replace('@name@', name)

    # write the new file
    with open(commands_folder+name+'.sh', 'w') as file:
        file.write(temp)
        file.close()
    
    batch.append('bash commands/'+name+'.sh')
    sbatch.append('sbatch commands/'+name+'.sh')

# write the batch file

check_mctal = False
mctal_folder = compute_folder+'output/mctal/'
with open(compute_folder+'batch.sh', 'w') as file:
    for line in batch:
        if check_mctal:
            if not os.path.isfile(mctal_folder+line.split()[1].split('/')[1].split('.')[0]+'.mctal'):
                file.write(line+'\n')
        else:
            file.write(line+'\n')
    file.close()

with open(compute_folder+'sbatch.sh', 'w') as file:
    for line in sbatch:
        if check_mctal:
            if not os.path.isfile(mctal_folder+line.split()[1].split('/')[1].split('.')[0]+'.mctal'):
                file.write(line+'\n')
        else:
            file.write(line+'\n')
    file.close()
