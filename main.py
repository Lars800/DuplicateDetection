import itertools
import math
import random
import numpy as np
import pandas as pd
import json
import CandidateGeneration as cg

f = open('TVs-all-merged.json')


# returns JSON object as
# a dictionary
data = json.load(f)
f.close()

dataframe = pd.json_normalize(data).transpose()
processed = pd.DataFrame(cg.dataPrep(dataframe)).transpose()

completeness = []
quality = []
frac = []
comp_final = []
qual_final = []
ok = True
for j in range(1,10):
    if ok:
        n_hash = 50
        n_bands = math.ceil(n_hash / j)
        n_hash = n_bands * j
        avg_can = []
        avg_comp = []
        avg_qual = []
        avg_comp_final = []
        avg_qual_final = []
        for i in range(5):
            print(j, i)
            # bootstrap
            is_train = np.zeros(len(processed[0]))
            for i in range(len(processed[0])):
                current = random.randint(0, len(processed[0])-1)
                if is_train[current] == 0:
                    is_train[current] = 1
            train_set = np.where(is_train==1)[0]
            test_set = np.where(is_train==0)[0]
            processed_train = processed.iloc[train_set, :].values.transpose().tolist()
            processed_test =  processed.iloc[test_set, :].values.transpose().tolist()

            weights = cg.train_methods(n_bands, n_bands, processed_train)

            candidates, lsh_encodings = cg.get_my_candidates(n_hash, n_bands, processed_test)
            duplicate_pairs = []
            is_duplicate = 0
            not_duplicate = 0
            for i in candidates:
                current = json.loads(i)
                pairs = list(itertools.combinations(current, 2))
                for pair in pairs:
                    duplicate_pairs.append(str(pair))
                    if processed_test[0][pair[0]] == processed_test[0][pair[1]]:
                        is_duplicate += 1
                    else:
                        not_duplicate += 1

            total_dups = cg.get_n_dups(processed_test[0])

            if len(candidates)==0 or total_dups==0:
                ok = False
                break
            avg_can.append(len(candidates)/(len(processed_test[0])* (len(processed_test[0])-1)))
            avg_qual.append( is_duplicate / len(candidates))
            avg_comp.append(is_duplicate / total_dups)

            found_duplicates = cg.generate_duplicates(candidates, lsh_encodings, processed_test, weights)
            correct_count = 0
            for pair in found_duplicates:
                current = json.loads(pair)
                if processed_test[0][current[0]] == processed_test[0][current[1]]:
                    correct_count +=1

            avg_comp_final.append(correct_count/total_dups)
            quality_c = 0
            if len(found_duplicates)>0:
                quality_c = correct_count/ len(found_duplicates)
            avg_qual_final.append(quality_c)




        comp_final.append(np.mean(np.array(avg_comp_final)))
        qual_final.append(np.mean(np.array(avg_qual_final)))
        frac.append(np.mean(np.array(avg_can)))
        completeness.append(np.mean(np.array(avg_comp)))
        quality.append(np.mean(np.array(avg_qual)))

