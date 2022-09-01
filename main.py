import pandas as pd

def fetch_data():
    data = pd.read_csv("elonmusk_tweets.csv")
    data_dict = {col: list(data[col]) for col in data.columns}

    return data_dict

def analyze_data(my_dict):

    print(type(my_dict))

if __name__=="__main__":
    analyze_data(fetch_data())