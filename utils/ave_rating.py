import numpy as np

def ave_rating(ratings):

    _label_key_ratings = ["rating_" + str(score) for score in range(1, 11)]
    ave_rating = np.sum(
        np.linspace(start=1, stop=10, num=10, endpoint=True) * \
        ratings[_label_key_ratings] / \
        np.sum(ratings[_label_key_ratings])
    )

    return ave_rating
