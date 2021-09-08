# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from pathlib import Path
import re
from numpy import arange
import seaborn as sns
import pathlib
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords
import math
from numpy import dot
from numpy.linalg import norm 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

# dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')

def task1():
    fd = open(datafilepath,)
    dt = json.load(fd)
    dt_lst = dt['teams_codes']
    dt_sorted = sorted(dt_lst)
    fd.close()
    return dt_sorted
    
def task2():
    # read the data
    fd = open(datafilepath,)
    dt = json.load(fd)
    fd.close()
    dt_club = dt['clubs']
    #extract the data and store them in lists
    lst1 = []
    lst2 = []
    for club in dt_club:
        lst1.append(club['goals_scored'])
        lst2.append(club['goals_conceded'])
    # Organise the data into one dataframe
    df = pd.DataFrame({'goals_scored_by_team': lst1, 'goals_scored_against_team': lst2})
    df.index = dt['teams_codes']
    df.index.name = 'teams_codes'
    df_out = df.sort_index()
    return df_out.to_csv('task2.csv')
      
def task3():
    # decide data struction and initialize
    score_dic = {}
    pat_score = r' (\d{1,2})-(\d{1,2})'
    
    # iteratively read the files and extract the data using regular expression 
    folder = Path(articlespath).rglob('*[0-9].txt')
    for article in folder:
        art = open(article,"r")
        content = art.read()
        art.close()
        match_scores = re.findall(pat_score, content)
        if match_scores != []:
            max_score = 0
            for score in match_scores:
                temp_score = int(score[0]) + int(score[1])
                max_score = max(temp_score, max_score)
            score_dic[article.name] = max_score                 
        else:
            score_dic[article.name] = 0      
    
    # sort and convert into panda series and 
    ds_sorted = pd.Series(score_dic).sort_index()                                               
    ds_sorted.name = 'total_goals'  
    ds_sorted.index.name = 'filename'
    return ds_sorted.to_csv('task3.csv')

def task4():
    # read and sort the data
    df = pd.read_csv('task3.csv') 
    df_sorted = df.sort_values(by = ['total_goals'])
    dt = df_sorted['total_goals']
    
    # plot
    boxplot_fig = plt.figure(num = 1, figsize = (10,6))
    
    plt.boxplot(df_sorted['total_goals'])
    plt.xlabel('articles')
    plt.ylabel('total goals')
    plt.title('maximum total goals in each article')
    
    # emphasize the outliers in the boxplot
    flier = dict(markerfacecolor='r', marker='o', markersize = 6)
    plt.boxplot(dt, flierprops = flier)
    
    return boxplot_fig.savefig('task4.png', bbox_inches='tight')
    
def task5():
    # extract the list of club names
    fd = open(datafilepath,)
    dt = json.load(fd)
    dt_lst = dt['participating_clubs']
    lst_sorted = sorted(dt_lst)
    fd.close()
    
    # initializion
    dic = dict.fromkeys(lst_sorted, 0)
    
    # iteratively read the files and extract the data using regular expression 
    folder = Path(articlespath).rglob('*[0-9].txt')
    for article in folder:
        art = open(article,"r")
        content = art.read()    
        for club_name in lst_sorted:
            if re.search(club_name, content):
                dic[club_name] += 1
            else:
                continue
    
    # plot the bar chart
    bar_fig = plt.figure(num =2, figsize = (10,8))
    clubs = list(dic.keys())
    times = list(dic.values())
    
    plt.bar(arange(len(times)), times)
    plt.xticks(arange(len(clubs)), clubs, rotation = 90,)
    
    plt.xlabel('club name') 
    plt.ylabel('number of mentions') 

    plt.title('Number of mentions for each club') 
    
    ds = pd.Series(dic)
    ds.name = 'number_of_mentions'  
    ds.index.name = 'club_name'
    return ds.to_csv('task5.csv'), bar_fig.savefig('task5.png', bbox_inches='tight')  

    
def task6():
    path = pathlib.Path('task5.csv')
    if path.exists():
        df_tsk5 = pd.read_csv('task5.csv')
    else:
        task5()
        df_tsk5 = pd.read_csv('task5.csv')

    club_name = df_tsk5['club_name']
    df = pd.DataFrame(data = 0, columns = club_name, index = club_name)
    length = len(club_name)
    folder = Path(articlespath).rglob('*[0-9].txt')
    for article in folder:
        art = open(article,"r")
        content = art.read()    
        art.close()
        for a in range(0, length):
            for b in range(a, length):
                if re.search(club_name[a], content) and re.search(club_name[b], content):
                    df.iloc[b,a] += 1
    for a in range(0, length):
        for b in range(a, length):        
            clb1_m = df_tsk5['number_of_mentions'][a]
            clb2_m = df_tsk5['number_of_mentions'][b]
            if clb1_m == 0 and clb2_m == 0:
                s = 1        
            else:
                s = df.iloc[b,a] * 2 / (clb1_m + clb2_m)
            df.iloc[b,a] = s        

    mask = np.zeros_like(df, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(num =3, figsize = (8,6))
    sns.heatmap(df,cmap='Greens',mask=mask, xticklabels = True)
    plt.title('Heatmap of similarity between clubs') 
    
    return plt.savefig('task6.png', bbox_inches='tight')


def task7():

    path2 = pathlib.Path('task2.csv')
    if path2.exists():
        df_tsk2 = pd.read_csv('task2.csv')
    else:
        task2()
        df_tsk2 = pd.read_csv('task2.csv')

    path1 = pathlib.Path('task5.csv')
    if path1.exists():
        df_tsk5 = pd.read_csv('task5.csv')
    else:
        task5()
        df_tsk5 = pd.read_csv('task5.csv')

    dic = {'goals_scored_by_team': df_tsk2.iloc[:,1], 'number_of_mentions': df_tsk5.iloc[:,1]}
    df = pd.DataFrame(dic)
    df.sort_values(by = 'goals_scored_by_team')

    plt.figure(num = 4, figsize = (8,6))
    plt.scatter(df.iloc[:,0], df.iloc[:,1], color='green')
    plt.ylabel("number_of_mentions")
    plt.xlabel("goals_scored_by_team")
    plt.grid(True)
    plt.title('scatterplot comparing the number of mentions and goals scored by each team') 
    
    return plt.savefig('task7.png', bbox_inches='tight')
    
def task8(filepath):
    fl = open(filepath,)
    dt = fl.read()
    fl.close()

    pt1 = r'[^a-zA-Z\n\t\s]'
    pt2 = r'\n|\t|\s\s+'

    dt = re.sub(pt1, ' ', dt)
    dt = re.sub(pt2, ' ', dt)
    dt = dt.lower()
    dt = nltk.word_tokenize(dt)

    stopWords = set(stopwords.words('english'))
    dt = [wd for wd in dt if wd not in stopWords and len(wd) > 1 ]
    return dt
    
def task9():
    dic = {}
    nm = []
    wd = []
    out = []
    folder = Path(articlespath).rglob('*[0-9].txt')
    for article in folder:
        nm.append(article.name)
        lst = task8(article)
        st = set(lst)
        dic[article.name] = {}
        for w in st:
            fq = lst.count(w)
            dic[article.name][w] = fq
        wd = wd + lst
    wd_set = set(wd)
    x = len(nm)
    y = len(wd_set)
    array = np.zeros(shape = (x, y), dtype = int)
    wd_lst = list(wd_set)
    for a in range(0, x):
        nm_ar = nm[a]
        for b in range(0, y):
            wd_ar = wd_lst[b]
            try:
                array[a][b] = dic[nm_ar][wd_ar]
            except:
                continue
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(array)
    arry_tfidf = tfidf.toarray()
    for a in range(0, x):
        nm_a = nm[a]
        for b in range(a+1, x):
            nm_b = nm[b]
            cosim = dot(arry_tfidf[a], arry_tfidf[b]) / (norm(arry_tfidf[a]) * norm(arry_tfidf[b]))
            out.append([nm_a, nm_b, cosim])
    df = pd.DataFrame(data = out, columns=('article1', 'article2', 'similarity'))
    df_sorted = df.sort_values(by = 'similarity', ascending=False)
    df_top10 = df_sorted.head(10)
    return df_top10.to_csv('task9.csv', index = None)