# -*- coding: utf-8 -*-

# A ML BASED ALGORITHM REPRESENTING THE FUNCTIONALITY OF CONTENT BASED FILTERING

"""
Created on Sat Jan 7 15:10:21 2023

@author: guneet
"""

#importing the libraries to create a sample database
import pandas as pd
from random import randint

#a function to generate the dataset based on various parameters
def generate_data(n_books = 3000, n_genres = 10, n_authors = 450, n_publishers = 50, n_readers = 30000, dataset_size = 100000):
    '''
    This function will generate a dataset with features associated to
    book data set. The dataset will have the following columns : 
        - book_id (String) : Unique identified for the book
        - book_rating (Integer) : A value between 0 and 10
        - reader_id (String) : Unique identifier for the user
        - book_genre (Integer) : An integer representing a genre for the book, 
                                 value is between 1 and 15, indicating that 
                                 there are 15 unique genres. Each book can only
                                 have 1 genre
        - author_id (String) : Unique identifier for the author of the book
        - num_pages (Integer) : Random value between 70 and 500
        - publisher_id (String) : A unique identifier for the publisher of the book
        - publish_year (Integer) : The year of book publishing
        - book_price (Integer) : The sale price of the book
        - text_lang (Integer) : The language of the book - returns an integer which 
                                is mapped to some language
        
    params:
        n_books (Integer) : The number of books you want the dataset to have
        n_genres (Integer) : Number of genres to be chosen from
        n_authors (Integer) : Number of authors to be generated
        n_publishers (Integer) : Number of publishers for the dataset
        n_readers (Integer) : Number of readers for the dataset
        dataset_size (Integer) : The number of rows to be generated 
        
    '''
    
    #assigning randomized values to a dataframe based on parameters.
    d = pd.DataFrame(
        {
            'book_id' : [randint(1, n_books) for _ in range(dataset_size)],
            'author_id' : [randint(1, n_authors) for _ in range(dataset_size)],
            'book_genre' : [randint(1, n_genres) for _ in range(dataset_size)],
            'reader_id' : [randint(1, n_readers) for _ in range(dataset_size)],
            'num_pages' : [randint(75, 700) for _ in range(dataset_size)],
            'book_rating' : [randint(1, 10) for _ in range(dataset_size)],
            'publisher_id' : [randint(1, n_publishers) for _ in range(dataset_size)],
            'publish_year' : [randint(2000, 2021) for _ in range(dataset_size)],
            'book_price' : [randint(1, 200) for _ in range(dataset_size)],
            'text_lang' : [randint(1,7) for _ in range(dataset_size)]
        }
    ).drop_duplicates()         #dropping duplicates along the way (if generated)
    return d                    #returung the data frame



d = generate_data(dataset_size = 100000)            #genetaing a dataset using the above function (rows in DF: 100000)
d.to_csv('data.csv', index = False)                 #converting the above generated dataset into a .CSV file


#importing libraries for data preprocessing and normalization
from numpy import dot
from numpy.linalg import norm 


#a function to normalize (scale) the generated data
def normalize(data):
    '''
    This function will normalize the input data to be between 0 and 1
    
    params:
        data (List) : The list of values you want to normalize
    
    returns:
        The input data normalized between 0 and 1
    '''
    min_val = min(data)
    if min_val < 0:
        data = [x + abs(min_val) for x in data]
    max_val = max(data)
    return [x/max_val for x in data]         #return a value between 0 and 1 for each element in the list


#function to apply one hot encoding to specified column and update it in the DF
def ohe(df, enc_col):
    '''
    This function will one hot encode the specified column and add it back
    onto the input dataframe
    
    params:
        df (DataFrame) : The dataframe you wish for the results to be appended to
        enc_col (String) : The column you want to OHE
    
    returns:
        The OHE columns added onto the input dataframe
    '''
    
    ohe_df = pd.get_dummies(df[enc_col])
    ohe_df.reset_index(drop = True, inplace = True)
    return pd.concat([df, ohe_df], axis = 1)            #returns the new data frame with the encoded column


#creating a class for the Content-Based Recommendation algorithm
class CBRecommend():
    
    #takes the dataframe as an input
    def __init__(self, df):
        self.df = df
    
    #funtion to calculate similarity between two vectors (dot product between them)
    def cosine_sim(self, v1,v2):
        '''
        This function will calculate the cosine similarity between two vectors
        '''
        return sum(dot(v1,v2)/(norm(v1)*norm(v2)))
    
    
    # a function to provide similar books based on the input book
    def recommend(self, book_id, n_rec):
        """
        df (dataframe): The dataframe
        book_id (string): Representing the book name
        n_rec (int): amount of rec user wants
        """
        
        # calculate similarity of input book_id vector w.r.t all other vectors
        inputVec = self.df.loc[book_id].values
        self.df['sim']= self.df.apply(lambda x: self.cosine_sim(inputVec, x.values), axis=1)

        # returns top n user specified books
        return self.df.nlargest(columns='sim',n=n_rec)

if __name__ == '__main__':
    # constants
    PATH = '../data/data.csv'   # the location of the dataframe (or the csv file so created)

    # import data
    df = pd.read_csv(PATH)  #importing the dataset from the location 
    
    
    """
    NOTE :  In our case we have created the df already using the randomized function 
            So, we'll be using the same below.
    """
    
    df = d

    # normalize the num_pages, ratings, price columns
    df['num_pages_norm'] = normalize(df['num_pages'].values)
    df['book_rating_norm'] = normalize(df['book_rating'].values)
    df['book_price_norm'] = normalize(df['book_price'].values)
    
    # One Hot Encoding on publish_year and genre
    df = ohe(df = df, enc_col = 'publish_year')
    df = ohe(df = df, enc_col = 'book_genre')
    df = ohe(df = df, enc_col = 'text_lang')
    
    # drop redundant columns
    cols = ['publish_year', 'book_genre', 'num_pages', 'book_rating', 'book_price', 'text_lang']
    df.drop(columns = cols, inplace = True)
    df.set_index('book_id', inplace = True)
    
    # ran on a sample as an example
    t = df.copy()
    cbr = CBRecommend(df = t)
    print(cbr.recommend(book_id = t.index[0], n_rec = 5))
