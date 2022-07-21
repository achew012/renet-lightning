import csv
import os
import json
import numpy as np
from tqdm import tqdm
import datetime

base_dir = "/storage/zjwu/MyProject/DSTA_Proj/Methods/DatasetConstructionGDELT/GDELT_0_and_txt_50/EG"


def pretransform(data_dir):
    actor_index, rel_index, complex_index = 0, 0, 0
    actor_set = {}
    rel_set = {}
    complex_set = {}
    result = {'train': [], 'eval': [], 'test': []}
    for event in tqdm(sorted(os.listdir(data_dir), key=lambda x: int(x[:8]))):
        year = int(event[:4])
        if year == 2020:
            target = "eval"
        elif year == 2021:
            target = "test"
        else:
            target = "train"
        fp = open(os.path.join(data_dir, event), mode='r')
        reader = csv.reader(fp, delimiter='\t')
        init = False
        if event not in complex_set.keys():
            complex_set[event] = complex_index
            complex_index += 1
        for line in reader:
            actor1 = line[0]
            actor2 = line[1]
            rel = line[2]
            timestamp = line[3]
            date = datetime.date(int(timestamp[:4]), int(timestamp[4:6]), int(timestamp[-2:]))
            if not init:
                start_date = date
                init = True
            if actor1 not in actor_set.keys():
                actor_set[actor1] = actor_index
                actor_index += 1
            if actor2 not in actor_set.keys():
                actor_set[actor2] = actor_index
                actor_index += 1
            if rel not in rel_set.keys():
                rel_set[rel] = rel_index
                rel_index += 1
            day = date - start_date
            newline = [actor_set[actor1], rel_set[rel], actor_set[actor2], day.days, complex_set[event]]
            result[target].append(newline)
        fp.close()
    for subset in result.keys():
        with open('data/' + subset + '.txt', mode='w') as f:
            writer = csv.writer(f, delimiter='\t')
            for line in result[subset]:
                writer.writerow(line)
    with open('data/entity2id.txt', mode='w') as f:
        writer = csv.writer(f, delimiter=' ')
        for key in actor_set.keys():
            writer.writerow([key, actor_set[key]])
    with open('data/relation2id.txt', mode='w') as f:
        writer = csv.writer(f, delimiter=' ')
        for key in rel_set.keys():
            writer.writerow([key, rel_set[key]])
    with open('data/cplxevent2id.txt', mode='w') as f:
        writer = csv.writer(f, delimiter=' ')
        for key in complex_set.keys():
            writer.writerow([key, complex_set[key]])


pretransform(base_dir)
