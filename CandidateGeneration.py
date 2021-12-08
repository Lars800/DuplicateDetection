import itertools
from random import shuffle
import re
import numpy as np
import pandas as pd
import json
import scipy.optimize
from sklearn.linear_model import LinearRegression
from kshingle import shingleseqs_list
from sklearn.metrics import f1_score
from scipy.cluster.hierarchy import linkage, fcluster


def dataPrep(dataframe):
    model_id = []
    features = []
    title = []
    brand = []
    size = []
    shop = []

    test_set = []
    for i in range(len(dataframe)):

        current_duplicates = []
        for j in range(len(dataframe[0][i])):
            current_duplicates.append(i + j)
            # Extract product info from dictionary
            model_id.append(dataframe[0][i][j]['modelID'])
            features.append(dataframe[0][i][j]['featuresMap'])

            shop.append(dataframe[0][i][j]['shop'])

            # extract title
            c_title = dataframe[0][i][j]['title'].lower()
            c_title = c_title.replace(r"\/|\(|\)", "")
            c_title = c_title.replace("'", "")
            c_title = re.sub(r'"| inch|-inch', 'inch', c_title)
            c_title = re.sub(r"\(|\)|/", '', c_title)
            title.append(c_title)
            # extract brand name
            brand_words = ['Brand Name:', 'Brand Name', 'Brand']
            for word in brand_words:
                if word in dataframe[0][i][j]['featuresMap']:
                    my_brand = dataframe[0][i][j]['featuresMap'][word]
                    if my_brand == 'JVC TV':
                        my_brand = 'JVC'
                    elif my_brand == 'LG Electronics':
                        my_brand = 'LG'
                    elif my_brand == 'Sceptre Inc.':
                        my_brand = 'Sceptre'
                    elif my_brand == 'Pansonic':
                        my_brand = 'Panasonic'
                    elif my_brand == 'Supersonic':
                        my_brand = 'SuperSonic'
                    elif my_brand == 'TOSHIBA':
                        my_brand = 'Toshiba'
                    brand.append(my_brand)
                    break
                elif word == 'Brand':
                    brand.append('UNKNOWN')

            size_words = ['Display Size', 'Diagonal Size',
                          'Screen Size Class', 'Diagonal Image Size:',
                          'Screen Size', 'Screen Size:', 'Screen Size (Measured Diagonally)']
            for word in size_words:
                if word in dataframe[0][i][j]['featuresMap']:
                    size_string = dataframe[0][i][j]['featuresMap'][word]
                    size_num = re.search(pattern="[0-9]{2}", string=size_string)
                    if size_num:
                        size.append(size_num.group())
                    else:
                        size.append(re.search(pattern="[0-9]", string=size_string).group())
                    break
                elif word == 'Screen Size:':
                    title_sizes = re.findall(
                        pattern=r'[0-9]{2}\\| [0-9]{2} | [0-9]{2}"| [0-9]{2}-| [0-9]{2}[iI] | [0-9]{2}[.]',
                        string=dataframe[0][i][j]['title'], )
                    title_sizes = list(set([re.sub(r"[^0-9]", "", i) for i in title_sizes]))
                    if len(title_sizes) != 0:
                        size.append(title_sizes[0])
                    else:
                        size.append('UNKNOWN')
                    break

    return model_id, title, brand, size, features, shop


def encode_vectors(final_vocab, processed_titles, processed_kpv):
    encoding_vectors = []
    max_len = 0
    for i in range(len(processed_titles)):
        current = processed_titles[i] + processed_kpv[i]
        max_len = max(len(current), max_len)
        vector = [1 if x in current else 0 for x in final_vocab]
        encoding_vectors.append(vector)
    return encoding_vectors


def signature_list(final_vocab, encoded_vectors, n_hash, n_bands):
    hashes = create_hash_table(len(final_vocab), n_hash)
    splitted_signatures = []
    full_signatures = []
    for encoded in encoded_vectors:
        signature = generate_signature(encoded, hashes, len(final_vocab))
        full_signatures.append(signature)
        split_sig = split_signature(signature, n_bands)
        splitted_signatures.append(split_sig)
    return splitted_signatures, full_signatures


def analize_titles(processed):
    title = processed[1]
    brands = processed[2]
    size = processed[3]
    features = processed[4]
    vocab = []
    processed_titles = []
    processed_kvp = []
    brand_set = set(brands)
    brand_set.add('Dynex')
    brand_set.add('Elite')

    for i in range(len(title)):
        title_words = title[i].split(' ')
        title_words = [i.upper() for i in title_words]
        if brands[i] == 'UNKNOWN':
            for brand in brand_set:
                if brand.upper() in title_words:
                    brands[i] = brand
                    break
        mw_title = re.findall(pattern=r"[a-zA-Z0-9]*[0-9]+[ˆ0-9,]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[ˆ0-9,]+[0-9]+[a-zA-Z0-9]*",
                              string=title[i], )
        processed_titles.append(mw_title)
        mw_kvp = []
        for feature in features[i]:
            model_words = re.findall(
                pattern=r"[a-zA-Z0-9]*[0-9]+[ˆ0-9,]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[ˆ0-9,]+[0-9]+[a-zA-Z0-9]*",
                string=features[i][feature])
            mw_kvp.extend(model_words)
        mw_kvp = list(set(mw_kvp))
        processed_kvp.append(mw_kvp)
        vocab.extend(mw_kvp)
        vocab.extend(mw_title)

    for x in set(vocab):
        vocab.remove(x)
    final_vocab = list(set(vocab))

    return final_vocab, processed_titles, processed_kvp, brands, size


def create_hash_table(size: int, nfunc: int):
    hash_list = []
    for i in range(1, nfunc + 1):
        sequence = list(range(1, size + 1))
        shuffle(sequence)
        hash_list.append(sequence)

    return hash_list


def generate_signature(encoded, hash_list, vocabsize):
    signature = []
    for my_hash in hash_list:
        for j in range(1, vocabsize + 1):
            ide = my_hash.index(j)
            if encoded[ide] == 1:
                signature.append(ide)
                break
    if len(signature) == 0:
        signature = [vocabsize + 1 for i in range(len(hash_list))]

    return signature


def split_signature(input, bands):
    assert len(input) % bands == 0
    l = int(len(input) / bands)
    splits = []
    for i in range(1, bands + 1):
        splits.append(input[i: i + l])
    return splits


def generate_cross_candidates(signatures_1, index_1, signatures_2, index_2):
    if len(signatures_1) == 0 or len(signatures_2) == 0:
        return [], {}
    bands = len(signatures_1[0])
    candidate_pairs = []
    obs_set = []

    for b in range(bands):
        known_values = []
        unkown_values = []
        for i in enumerate(signatures_1):
            known_values.append(str(signatures_1[i[0]][b]))
        for i in enumerate(signatures_2):
            unkown_values.append(str(signatures_2[i[0]][b]))
        unkown_set = list(set(unkown_values))
        known_array = np.array(known_values)
        unknown_array = np.array(unkown_values)

        for x in unkown_set:
            known_mapped = np.where(known_array == x)[0]
            unknown_mapped = np.where(unknown_array == x)[0]
            for k in known_mapped:
                for u in unknown_mapped:
                    candidate_pairs.append(str([index_1[k], index_2[u]]))
                    obs_set.extend([index_1[k], index_2[u]])
    return candidate_pairs


def jaccard(left, right):
    left = set(left)
    right = set(right)
    inter = left.intersection(right)
    union = left.union(right)
    if len(union) == 0:
        return 0
    sim = len(inter) / len(union)
    return sim


def jaccard_w(left, right, frequency):
    left = set(left)
    right = set(right)
    inter = left.intersection(right)
    union = left.union(right)
    w1 = 0
    w2 = 0
    for two in union:
        w2 += 1 / frequency[two]
        if two in inter:
            w1 += 1 / frequency[two]
    if len(inter) > 0:
        return w1 / w2
    else:
        return 0


def get_mw(string_in):
    split = re.findall(pattern=r"[a-zA-Z0-9]*[0-9]+[ˆ0-9,]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[ˆ0-9,]+[0-9]+[a-zA-Z0-9]*",
                       string=string_in.lower())
    return split


def extract_idef(title1, title2):
    sp1 = re.findall(
        pattern=r"[a-zA-Z0-9]+-[a-zA-Z0-9]+[0-9]+|[a-zA-Z0-9]*[a-zA-Z]+[0-9]+[a-zA-Z][a-zA-Z0-9]*|[a-zA-Z0-9]*[0-9]+[a-zA-Z]+[0-9]+[a-zA-Z0-9]*",
        string=title1.lower())
    sp2 = re.findall(
        pattern=r"[a-zA-Z0-9]+-[a-zA-Z0-9]+[0-9]+|[a-zA-Z0-9]*[a-zA-Z]+[0-9]+[a-zA-Z][a-zA-Z0-9]*|[a-zA-Z0-9]*[0-9]+[a-zA-Z]+[0-9]+[a-zA-Z0-9]*",
        string=title2.lower())
    if len(sp1) == 0 or len(sp2) == 0:
        return 0
    if sp1 == sp2:
        return 1
    return -1


def generate_duplicates(candidates, lsh_encodings, processed, weights):
    title = processed[1]
    features = processed[4]
    full_signatures = lsh_encodings[1]
    idf = lsh_encodings[4]

    duplicates = []
    candidates = list(set(candidates))
    distance_mat = np.full((len(title), len(title)), 25, dtype=float)
    for i in enumerate(candidates):
        current = sorted(json.loads(i[1]))
        id_check = extract_idef(title[current[0]], title[current[1]])
        if id_check == 1:
            distance_mat[current[0]][current[1]] = 0
        elif id_check == -1:
            distance_mat[current[0]][current[1]] = 5
        else:
            weighted_total = 1 - jaccard_w(full_signatures[current[0]], full_signatures[current[1]], idf)
            j_sim = 0
            n = 0
            unmatched_left = []
            unmatched_right = []
            for j in features[current[0]]:
                if j in features[current[1]]:
                    s1 = shingleseqs_list(features[current[0]][j].lower(), klist=[3])
                    s2 = shingleseqs_list(features[current[1]][j].lower(), klist=[3])
                    j_c = jaccard(s1[0], s2[0])
                    if j_c > 0.80:
                        j_sim += j_c
                        n += 1
                    else:
                        n += 0.5
                else:
                    cmw = get_mw(features[current[0]][j])
                    unmatched_left.extend(cmw)
            if n > 0:
                j_sim /= n
            for j in features[current[1]]:
                if j not in features[current[0]]:
                    cmw = get_mw(features[current[1]][j])
                    unmatched_right.extend(cmw)
            j_sim = 1 - j_sim
            unmatched_score = 1 - jaccard(unmatched_right, unmatched_left)

            distance_mat[current[0]][current[1]] = max(weights[3] + weights[0]* j_sim + weights[1]*unmatched_score + weights[2]*weighted_total, 0)

    if len(candidates) > 1:
        flat_dist = []
        for i in range(len(distance_mat)):
            for j in range(i + 1, len(distance_mat)):
                flat_dist.append(distance_mat[i][j])
        flat_dist = np.array(flat_dist)
        clus_model = linkage(flat_dist, method='single', metric='euclidean')
        assignments = fcluster(clus_model, t=0.1, criterion='distance')
        for clus in range(1, np.max(assignments)):
            clustr = np.where(assignments == clus)[0]
            if len(clustr) > 1:
                duplicates.append(str(list(clustr)))


    return duplicates


def split_data_within(data_in, current_features, set_lists, max_level, level, lsh_endodings, processed):
    candidates = []
    if level < max_level:
        feature_set = set_lists[level]
        for feature in feature_set:
            data_out = data_in[data_in[level] == feature]
            if feature == 'UNKNOWN':
                data_out2 = data_in[data_in[level] != feature]

                cc = split_data_cross(data_out, data_out2, set_lists, max_level, level + 1, lsh_endodings,
                                            processed)
                candidates.extend(cc)
            if len(data_out) > 2:
                next_features = current_features.copy()
                next_features.append(feature)

                cc = split_data_within(data_out, next_features, set_lists, max_level, level + 1, lsh_endodings,
                                             processed)
                candidates.extend(cc)
    else:
        undesired = []
        for shop in set(data_in[max_level]):
            undesired.append(shop)
            ds1 = data_in[data_in[max_level] == shop]
            data_in = data_in[data_in[max_level] != shop]
            if len(ds1) > 0 and len(data_in) > 0:
                current_candidates = generate_cross_candidates(list(ds1[max_level + 1]),
                                                                            list(ds1[max_level + 2]),
                                                                            list(data_in[max_level + 1]),
                                                                            list(data_in[max_level + 2]))
                candidates.extend(current_candidates)
    if len(candidates)>0:
        candidates = list(set(candidates))
    return candidates


def split_data_cross(data_in_1, data_in_2, set_lists, max_level, level, lsh_endodings, processed):
    candidates = []
    if level < max_level:
        feature_set = set_lists[level]
        bool_uk = False
        if 'UNKNOWN' in feature_set:
            uk_1 = data_in_1[data_in_1[level] == 'UNKNOWN']
            uk_2 = data_in_2[data_in_2[level] == 'UNKNOWN']
            if len(uk_1) > 0 and len(uk_2) > 0:
                cc = split_data_cross(uk_1, uk_2, set_lists, max_level, level + 1, lsh_endodings, processed)
                candidates.extend(cc)
                bool_uk = True
            feature_set.remove('UNKNOWN')
        for feature in feature_set:
            out_1 = data_in_1[data_in_1[level] == feature]
            out_2 = data_in_2[data_in_2[level] == feature]
            if len(out_1) > 0 and len(out_2) > 0:
                cc = split_data_cross(out_1, out_2, set_lists, max_level, level + 1, lsh_endodings, processed)
                candidates.extend(cc)

            if len(out_2) > 0 and bool_uk:
                cc = split_data_cross(out_1, data_in_1[data_in_1[level] == 'UNKNOWN'], set_lists, max_level,
                                            level + 1,
                                            lsh_endodings, processed)
                candidates.extend(cc)

    else:
        for shop in set(data_in_1[max_level]):
            ds1 = data_in_1[data_in_1[max_level] == shop]
            ds2 = data_in_2[data_in_2[max_level] != shop]
            if len(ds1) > 0 and len(ds2) > 0:
                current_candidates = generate_cross_candidates(list(ds1[max_level + 1]),
                                                                            list(ds1[max_level + 2]),
                                                                            list(ds2[max_level + 1]),
                                                                            list(ds2[max_level + 2]))
                candidates.extend(current_candidates)

    if len(candidates)>0:
        candidates = list(set(candidates))
    return candidates




def get_my_candidates(n_hash, n_bands, processed):
    shop = processed[5]
    index = list(range(len(shop)))
    vocab, processed_title, processed_kpv, brands, size = analize_titles(processed)

    encoded_vectors = encode_vectors(vocab, processed_title, processed_kpv)
    splitted_signatures, full_signatures = signature_list(vocab, encoded_vectors, n_hash, n_bands)

    idf = np.zeros(len(vocab) + 2)
    for sig in full_signatures:
        for word in sig:
            idf[word] += 1
    idf[np.where(idf == 0)[0]] = 1

    lsh_endodings = [vocab, full_signatures, processed_title, processed_kpv, idf]

    my_data = pd.DataFrame([brands, size, shop, splitted_signatures, index]).transpose()
    brand_set = set(brands)
    size_set = set(size)
    set_list = [brand_set, size_set]
    candidates = split_data_within(my_data, [], set_list, len(set_list), 0, lsh_endodings, processed)
    return candidates, lsh_endodings


def accuracy_thres(threshold, is_not_dup, applied_distance ):
    pred = []
    for obs in enumerate(is_not_dup):
        if  applied_distance[obs[0]] > threshold:
            pred.append(1)
        elif  applied_distance[obs[0]] <= threshold:
            pred.append(1)

    f1 = f1_score(is_not_dup, pred)
    return -f1

def train_methods(n_hash, n_bands, processed):
    shop = processed[5]
    index = list(range(len(shop)))
    vocab, processed_title, processed_kpv, brands, size = analize_titles(processed)

    encoded_vectors = encode_vectors(vocab, processed_title, processed_kpv)
    splitted_signatures, full_signatures = signature_list(vocab, encoded_vectors, n_hash, n_bands)

    idf = np.zeros(len(vocab) + 2)
    for sig in full_signatures:
        for word in sig:
            idf[word] += 1
    idf[np.where(idf == 0)[0]] = 1

    lsh_endodings = [vocab, full_signatures, processed_title, processed_kpv, idf]

    my_data = pd.DataFrame([brands, size, shop, splitted_signatures, index]).transpose()
    brand_set = set(brands)
    size_set = set(size)
    set_list = [brand_set, size_set]
    candidates = split_data_within(my_data, [], set_list, len(set_list), 0, lsh_endodings, processed)
    set_candidates = set(candidates)
    distances = list(calc_dist(set_candidates, processed))
    is_no_dup = []
    for candidate in set_candidates:
        current = json.loads(candidate)
        if processed[0][current[0]] == processed[0][current[1]]:
            is_no_dup.append(0)
        else:
            is_no_dup.append(1)
    estimator = LinearRegression().fit(np.array(distances).transpose(), np.array(is_no_dup))

    distances = np.array(distances)
    applied_distance = estimator.intercept_ + np.sum(estimator.coef_ * np.array(distances).transpose(), axis=1)
    threshold = scipy.optimize.minimize_scalar(accuracy_thres, args=(is_no_dup, applied_distance), bounds=(0, 1), tol=1e-6)
    weights = list(estimator.coef_)
    weights.append(estimator.intercept_)
    weights.append(threshold.x)
    return  weights




def calc_dist(candidates, processed):
    features = processed[4]
    unmatched_s = []
    feature_sim = []
    title_J = []

    for i in enumerate(candidates):
        current = sorted(json.loads(i[1]))
        j_sim = 0
        n = 0
        unmatched_left = []
        unmatched_right = []
        for j in features[current[0]]:
            if j in features[current[1]]:
                s1 = shingleseqs_list(features[current[0]][j].lower(), klist=[3])
                s2 = shingleseqs_list(features[current[1]][j].lower(), klist=[3])
                j_c = jaccard(s1[0], s2[0])
                j_sim += j_c
                n += 1
            else:
                cmw = get_mw(features[current[0]][j])
                unmatched_left.extend(cmw)
        if n > 0:
            j_sim /= n
        feature_sim.append(1 - j_sim)
        for j in features[current[1]]:
            if j not in features[current[0]]:
                cmw = get_mw(features[current[1]][j])
                unmatched_right.extend(cmw)
        MW1 = get_mw(processed[0][current[0]])
        MW2 = get_mw(processed[0][current[1]])
        title_sim = jaccard(MW1, MW2)
        title_J.append(1 - title_sim)
        unmatched_score = jaccard(unmatched_right, unmatched_left)
        unmatched_s.append(1 - unmatched_score)

    return title_J, unmatched_s, feature_sim


def get_n_dups(processed):
    model_id_set = set(processed)
    model_id_array = np.array(processed)
    duplicate_set = []
    for id in model_id_set:
        current_id = np.where(model_id_array == id)[0]
        if len(current_id) > 1:
            dups = list(itertools.combinations(current_id, 2))
            for dup in dups:
                duplicate_set.append(str(list(dup)))

    return len(duplicate_set)
