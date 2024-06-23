import pandas as pd
from pathlib import Path
import csv


dp = Path('./data/')
for d in dp.iterdir(): # subdirectories 'phase_k'
    if d.is_dir():
        print('dir: ', d)
        # initialize list for subdirectory. Each item is a dictionary
        # corresponding to a row in a pandas dataframe.
        dc_list = [] 
        print(d.name)
        # Get all of the npy file root names
        base_names = set()
        all_npy_files = [f for f in d.glob('**/*') if f.suffix == '.npy']
        for f in all_npy_files:
            base_names.add('_'.join(f.name.split('_')[:-1]))
                # print(base_names)
        print(base_names)
        # Now find all files with a matching base_name and add them to dc_list
        for base_name in base_names:
            dc = {}
            for f in all_npy_files:
                if base_name in f.name:
                    kv = base_name.split('_')[-1]
                    typev = f.name.split('_')[-1].split('.')[0]
                    base = '_'.join(base_name.split('_')[:-1])
                    fd = f.parent
                    # Now add the corresponding model files
                    model_files = fd.glob('*.xml')
                    scenef, combof, humanf = '', '', ''
                    for mf in model_files:
                        if 'scene' in mf.name:
                            scenef = str(mf.name)
                        elif 'and' in mf.name:
                            combof = str(mf.name)
                        elif 'humanoid' in mf.name:
                            humanf = str(mf.name)
                    # Finally, append a dictionary to the list
                    dc_list.append({'base': base, 'type': typev, 'k': kv,
                                    'dir': str(f.parent),
                                    'filename': f.name, 'scene_xml': scenef,
                                    'humanoid_xml': humanf,
                                    'combo_xml': combof})
        # Special case for if there are no .npy files. In this case, there are
        # only model files
        if len(base_names) == 0:
            model_files = d.glob('*.xml')
            scenef, combof, humanf = '', '', ''
            for mf in model_files:
                if 'scene' in mf.name:
                    scenef = str(mf.name)
                elif 'and' in mf.name:
                    combof = str(mf.name)
                elif 'humanoid' in mf.name:
                    humanf = str(mf.name)
            dc_list.append({'dir': str(d),
                    'scene_xml': scenef,
                    'humanoid_xml': humanf,
                    'combo_xml': combof})
        # Now export to csv, using pandas DataFrame
        # df = pd.DataFrame(dc_list)
        with open(d/'dataset.csv', 'w', newline='') as f:
            fc = csv.DictWriter(f, fieldnames=dc_list[0].keys())
            fc.writeheader()
            fc.writerows(dc_list)
            # df.to_csv(f)
        # Make a dictionary entry for each base name which includes every
        # filepath that matches the base name

