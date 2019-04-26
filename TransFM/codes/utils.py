import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split


class DataGenerator(object):
    def __init__(self, filePath, tagFilePath=None, sorted=False, encoding='utf8'):
        self.df_temp = pd.read_csv(filePath, encoding=encoding)
        self.df_tag = pd.read_csv(tagFilePath, encoding=encoding)
        self.useTag = tagFilePath == None
        if self.useTag:
            self.df = self.generate_df_with_tag()
        else:
            self.df = self.df_temp
        self.df_total = self.generate_df_with_tag()
        self.sorted = sorted
        self.rows_num = self.df.shape[0]
        self.cols_num = self.df.shape[1]

    def generate_df_with_tag(self):
        '''
        Generate dataframe with tag
        :return: dataframe
        '''
        df = self.df_temp
        if self.useTag:
            df_tag = self.df_tag[['movieId', 'tag']]
            df_tag.reset_index(inplace=True)
            df_tag.columns = ['tagId', 'movieId', 'tag']
            df_total = pd.merge(df, df_tag, on=['movieId'], how='left')
            df_user_dummies = pd.get_dummies(df_total['userId'], prefix='user')
            df_movie_dummies = pd.get_dummies(df_total['movieId'], prefix='movie')
            df_tag_dummies = pd.get_dummies(df_total['tagId'], prefix='tag')
            df_total = pd.concat([df, df_user_dummies, df_movie_dummies,df_tag_dummies], axis=1)
            # df_total.drop(['userId', 'movieId', 'tagId'], inplace=True)
        else:
            df_user_dummies = pd.get_dummies(df['userId'], prefix='user')
            df_movie_dummies = pd.get_dummies(df['movieId'], prefix='movie')
            df_total = pd.concat([df, df_user_dummies, df_movie_dummies], axis=1)
            # df_total.drop(['userId', 'movieId'], axis=1, inplace=True)
        return df_total

    def generate_data_batch(self, batch_size=100, duplicate=True, sorted_by_time=True):
        '''
        Generate batched data with num of batch_size
        :param batch_size: The length of each batch
        :param duplicate: Whether can pick the same element or not
        :param sorted_by_time: Whether sorted specific data by timestamp or not
        :return: batched data with specific size(default 100)
        '''
        choices = None
        if duplicate:
            choices = np.random.choice(self.rows_num, batch_size)
        else:
            choices = random.shuffle(np.arange(0, self.rows_num))[:batch_size]
        batch_data = self.df_total.iloc[choices, :]
        if sorted_by_time:
            batch_data = batch_data.sort_values(['userId', 'timestamp'])
        target = batch_data[['rating']]
        features = batch_data.drop(['rating'], axis=1)
        return features.values.reshape((features.shape[0], features.shape[1], 1, 1)), target.values


if __name__ == '__main__':
    dataGen = DataGenerator(filePath='data/MovieLen/ratings_tiny.csv', tagFilePath='data/MovieLen/tags.csv')
    batch_features, batch_targets = dataGen.generate_data_batch()
    print(batch_features[:5])