import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

def fetch_data():
    data = pd.read_csv("elonmusk_tweets.csv",parse_dates=True)
    data_dict = {col: list(data[col]) for col in data.columns}

    return data_dict

def analyze_data(my_dict):

    #print(type(my_dict['text']))

    analyzer = SentimentIntensityAnalyzer()
    
    sentiments=[]

    #sentiment analysis
    for i in my_dict['text']:
        text = str(i)
        ps = analyzer.polarity_scores(text)
        sentiments.append({'text':text, 'compound':ps['compound']})
    #sentiment df
    tweetdf = pd.DataFrame(sentiments)
    #print(tweetdf)
    #Pie chart code
    count=dict(positive=0,neutral=0,negative=0)

    for i in tweetdf['compound']:
        if i >= 0.05:
                count['positive']+=1
        elif i <= -0.05:
                 count['negative']+=1
        else:
                 count['neutral']+=1


    plt.pie(count.values(), labels=count.keys())

    plt.axis('equal')
    plt.savefig('pie_chart.png')
    plt.close('all')

    #line graph code 
    #print(my_dict['created_at'])

    a = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in my_dict['created_at']]

    x = matplotlib.dates.date2num(a)

    formatter = matplotlib.dates.DateFormatter('%Y')

    figure = plt.figure()

    axes = figure.add_subplot(1, 1, 1)
    axes.xaxis.set_major_formatter(formatter)

    axes.plot(x, [sentiments[i]['compound'] for i in range(len(sentiments))])
    # set axis titles
    plt.xlabel("Time")
    plt.ylabel("Scores")
    # set chart title
    plt.title("Elon Musk tweet sentiment over time")
    plt.savefig('line_graph.png')


if __name__=="__main__":
    analyze_data(fetch_data())

