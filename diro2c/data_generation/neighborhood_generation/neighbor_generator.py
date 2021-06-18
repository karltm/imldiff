from data_generation.neighborhood_generation.gpdatagenerator import *
from data_generation.neighborhood_generation.modified_gpdatagenerator import *
from data_generation.diff_dataset_builder import build_dict_dataset_diff
from data_generation.distance_functions import *
from data_generation.helper import *


def get_genetic_neighborhood(x, blackbox1, blackbox2, dataset, X_to_recognize_diff, diff_classifier_method,
                             neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, max_steps=1, is_unique=True):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(
        X_to_recognize_diff, dataset['columns'], class_name, discrete, continuous, 1000)

    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z1 = generate_data(x, feature_values, blackbox1, discrete_no_class, continuous, class_name, idx_features,
                       distance_function, neigtype=neigtype, population_size=population_size, halloffame_ratio=0.1,
                       alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10, return_logbook=False, max_steps=max_steps, is_unique=is_unique)

    Z2 = generate_data(x, feature_values, blackbox2, discrete_no_class, continuous, class_name, idx_features,
                       distance_function, neigtype=neigtype, population_size=population_size, halloffame_ratio=0.1,
                       alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10, return_logbook=False, max_steps=max_steps, is_unique=is_unique)

    # zy1 = blackbox1.predict(Z1)
    # zy2 = blackbox2.predict(Z2)

    # print("zy1 predicts: ", np.unique(zy1, return_counts=True))
    # print("zy2 predicts: ", np.unique(zy2, return_counts=True))

    Z = np.concatenate((Z1, Z2))

    if is_unique:
        Z = np.unique(Z, axis=0)

    #print("generated len(Z): ", len(Z))

    # np.savetxt("unique_test.csv", Z, delimiter=",")

    dict_dataset_diff = build_dict_dataset_diff(
        blackbox1, blackbox2, Z, dataset, diff_classifier_method)

    return dict_dataset_diff


def get_modified_genetic_neighborhood(x, blackbox1, blackbox2, dataset, X_to_recognize_diff, diff_classifier_method,
                                      neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, max_steps=1, is_unique=True):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(
        X_to_recognize_diff, dataset['columns'], class_name, discrete, continuous, 1000)

    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    try:
        discrete_no_class.remove(class_name)
    except ValueError:
        print("Warning: discrete_no_class --> class_name can't be found to remove!")

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z = generate_modified_data(x, feature_values, blackbox1, blackbox2, diff_classifier_method, discrete_no_class, continuous, class_name, idx_features,
                               distance_function, neigtype=neigtype, population_size=population_size, halloffame_ratio=0.1,
                               alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10, return_logbook=False, max_steps=max_steps, is_unique=is_unique)

    # zy1 = blackbox1.predict(Z1)
    # zy2 = blackbox2.predict(Z2)

    # print("zy1 predicts: ", np.unique(zy1, return_counts=True))
    # print("zy2 predicts: ", np.unique(zy2, return_counts=True))

    #Z = np.concatenate((Z1, Z2))

    if is_unique:
        Z = np.unique(Z, axis=0)

    #print("generated len(Z): ", len(Z))

    # np.savetxt("unique_test.csv", Z, delimiter=",")

    dict_dataset_diff = build_dict_dataset_diff(
        blackbox1, blackbox2, Z, dataset, diff_classifier_method)

    return dict_dataset_diff


def get_closed_real_data(x, blackbox1, blackbox2, dataset, X_test_dataset, diff_classifier_method, k=1000, is_unique=True):
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    continuous = dataset['continuous']
    idx_features = dataset['idx_features']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    neig_indexes1 = _get_closest_diffoutcome(X_test_dataset, x, discrete_no_class, continuous, class_name, idx_features,
                                             blackbox1, distance_function, k=k)

    neig_indexes2 = _get_closest_diffoutcome(X_test_dataset, x, discrete_no_class, continuous, class_name, idx_features,
                                             blackbox2, distance_function, k=k)

    Z1 = X_test_dataset[neig_indexes1]
    Z2 = X_test_dataset[neig_indexes2]

    Z = np.concatenate((Z1, Z2))
    if is_unique:
        Z = np.unique(Z, axis=0)

    #print("generated len(Z): ", len(Z))

    dict_dataset_diff = build_dict_dataset_diff(
        blackbox1, blackbox2, Z, dataset, diff_classifier_method)

    return dict_dataset_diff


# private
def _get_closest_diffoutcome(X_full_dataset, x, discrete, continuous, class_name, idx_features, blackbox, distance_function,
                             k=250, diff_out_ratio=0.3):
    distances = list()
    distances_0 = list()
    idx0 = list()
    distances_1 = list()
    idx1 = list()
    #Z, _ = label_encode(df, discrete, label_encoder)
    #Z = X_full_dataset.iloc[:, X_full_dataset.columns != class_name].values

    x = {idx_features[i]: val for i, val in enumerate(x)}

    idx = 0
    for z in X_full_dataset:
        z1 = {idx_features[i]: val for i, val in enumerate(z)}
        d = distance_function(x, z1, discrete, continuous, class_name)
        distances.append(d)
        if blackbox.predict(z.reshape(1, -1))[0] == 0:
            distances_0.append(d)
            idx0.append(idx)
        else:
            distances_1.append(d)
            idx1.append(idx)
        idx += 1

    idx0 = np.array(idx0)
    idx1 = np.array(idx1)

    all_indexs = np.argsort(distances).tolist()[:k]
    indexes0 = list(idx0[np.argsort(distances_0).tolist()[:k]])
    indexes1 = list(idx1[np.argsort(distances_1).tolist()[:k]])

    if 1.0 * len(set(all_indexs) & set(indexes0)) / len(all_indexs) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + \
            indexes0[:int(k * diff_out_ratio)]
    elif 1.0 * len(set(all_indexs) & set(indexes1)) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + \
            indexes1[:int(k * diff_out_ratio)]
    else:
        final_indexes = all_indexs

    return final_indexes
