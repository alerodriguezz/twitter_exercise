import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_data():
    data = pd.read_csv("elonmusk_tweets.csv")
    data_dict = {col: list(data[col]) for col in data.columns}

    return data_dict

def analyze_data(my_dict):

    #print(type(my_dict['text']))
    analyzer = SentimentIntensityAnalyzer()

    temp=[]

    for i in my_dict['text']:
        text = str(i)
        ps = analyzer.polarity_scores(text)
        temp.append({'text':text, 'compound':ps['compound']})
    
    tweetdf = pd.DataFrame(temp)
    
    count=dict(positive=0,neutral=0,negative=0)

    for i in tweetdf['compound']:
        if i >= 0.05:
                count['positive']+=1
        elif i <= -0.05:
                 count['negative']+=1
        else:
                 count['neutral']+=1


   

    # Plot
    plt.pie(count.values(), labels=count.keys())

    plt.axis('equal')
    plt.savefig('sample.png')


if __name__=="__main__":
    analyze_data(fetch_data())

