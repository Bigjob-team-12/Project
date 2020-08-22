import pandas as pd
import os
from dog_breed_similarity_comparison import load_data, cos_sim, euc_sim, pearson

def load_similar_images(file_name, func = cos_sim, N = 10):
    '''
    input image와 유사한 image 10개 추출
    :param file_name:
    :param path:
    :return: file list, image list
    '''
    global result

    # input image와 cosine 유사도가 높은 10개의 file list
    file_lst = data.apply(lambda x: func(x, data.loc[file_name]), axis=1).sort_values(ascending=False)[1:N+2].index
    class_lst = [_.split('/')[0] for _ in file_lst]

    tmp_class = file_name.split('/')[0]
    result.extend([(tmp_class, _) for _ in class_lst])
def similarity_evaluation(path, similarity_func, N = 10):
    # load test data
    dir_lists = os.listdir(path)
    print(dir_lists)

    for class_lst in dir_lists:
        class_path = os.path.join(path, class_lst)

        file_lists = os.listdir(class_path)
        for file_lst in file_lists:

            print(class_lst + '/' + file_lst)
            file_name = class_lst + '/' + file_lst
            load_similar_images(file_name, similarity_func, N)


if __name__ == '__main__':
    # data load
    data = load_data()

    results_by_similarity = []

    path = '../../../_db/data/model_data/input/dog_data/ours_dog/test'

    N = 50

    for similarity_func in [cos_sim, euc_sim, pearson]:
        result = []
        similarity_evaluation(path, similarity_func, N)

        results_by_similarity.append([str(similarity_func).split()[1], round(pd.DataFrame(result).apply(lambda x : x[0] == x[1],axis = 1).mean(),3)])
        print(len(result))

    print(results_by_similarity)