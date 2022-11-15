# import a library nltk and a function from it
import nltk
from nltk.corpus import stopwords
# import a library re
import re
# import a library string
import string
# import a library pandas
import pandas as pd
# import a library numpy
import numpy as np
# import a library matplotlib
import matplotlib.pyplot as plt
# write a function to remove punctuation
def remove_punctuation(text):
    # remove punctuation
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    # return the result
    return text_nopunct
# write a function to tokenize
def tokenize(text):
    # split the text by space
    tokens = re.split('\W+', text)
    # return the result
    return tokens
# write a function to remove stopwords
def remove_stopwords(tokenized_list):
    # remove stopwords
    text = [word for word in tokenized_list if word not in stopwords.words('english')]
    # return the result
    return text
# write a function to stem
def stemming(tokenized_text):
    # stem the text
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in tokenized_text]
    # return the result
    return text
# write a function to lemmatize
def lemmatizing(tokenized_text):
    # lemmatize the text
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    # return the result
    return text
# write a function to clean the text
def clean_text(text):
    # remove punctuation
    text = remove_punctuation(text)
    # tokenize
    text = tokenize(text.lower())
    # remove stopwords
    text = remove_stopwords(text)
    # stem
    text = stemming(text)
    # lemmatize
    text = lemmatizing(text)
    # return the result
    return text
# write a function to count the words
def count_words(text):
    # count the words
    count = len(text)
    # return the result
    return count
# write a function to count the unique words
def count_unique_words(text):
    # count the unique words
    count = len(set(text))
    # return the result
    return count
# write a function to count the punctuation
def count_punctuation(text):
    # count the punctuation
    count = len([char for char in text if char in string.punctuation])
    # return the result
    return count
# write a function to count the stopwords
def count_stopwords(text):
    # count the stopwords
    count = len([word for word in text if word in stopwords.words('english')])
    # return the result
    return count
# write a function to count the numbers
def count_numbers(text):
    # count the numbers
    count = len([word for word in text if word.isdigit()])
    # return the result
    return count
# write a function to count the uppercase words
def count_uppercase(text):
    # count the uppercase words
    count = len([word for word in text if word.isupper()])
    # return the result
    return count
# write a function using pandas to read the csv file and return a dataframe
def read_csv_file(filename):
    # read the csv file
    df = pd
    # return the result
    return df
# write a function to plot the bar chart
def plot_bar_chart(df, x, y, title, xlabel, ylabel, color):
    # plot the bar chart
    df.plot.bar(x, y, title = title, color = color)
    # set the x label
    plt.xlabel(xlabel)
    # set the y label
    plt.ylabel(ylabel)
    # show the plot
    plt.show()
# write a function to plot the histogram
def plot_histogram(df, x, title, xlabel, ylabel, color):
    # plot the histogram
    df.plot.hist(x, title = title, color = color)
    # set the x label
    plt.xlabel(xlabel)
    # set the y label
    plt.ylabel(ylabel)
    # show the plot
    plt.show()
# write a function to plot the boxplot
def plot_boxplot(df, x, y, title, xlabel, ylabel, color):
    # plot the boxplot
    df.plot.box(x, y, title = title, color = color)
    # set the x label
    plt.xlabel(xlabel)
    # set the y label
    plt.ylabel(ylabel)
    # show the plot
    plt.show()
# write a function to plot the scatterplot
def plot_scatterplot(df, x, y, title, xlabel, ylabel, color):
    # plot the scatterplot
    df.plot.scatter(x, y, title = title, color = color)
    # set the x label
    plt.xlabel(xlabel)
    # set the y label
    plt.ylabel(ylabel)
    # show the plot
    plt.show()
# write a function to plot the pie chart
def plot_pie_chart(df, x, title, color):
    # plot the pie chart
    df.plot.pie(x, title = title, colors = color)
    # show the plot
    plt.show()
# write a function to plot the line chart
def plot_line_chart(df, x, y, title, xlabel, ylabel, color):
    # plot the line chart
    df.plot.line(x, y, title = title, color = color)
    # set the x label
    plt.xlabel(xlabel)
    # set the y label
    plt.ylabel(ylabel)
    # show the plot
    plt.show()
