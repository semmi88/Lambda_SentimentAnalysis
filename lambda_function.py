from __future__ import print_function

import json
import boto3
import re
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


comprehendClient = boto3.client('comprehend')
s3Client = boto3.client('s3')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    asins = event['asins']
    reviewOutput = {}
    for asin in asins:
        asinData = {}
        reviewIndex = 0;
        url = "https://www.amazon.com/product-reviews/" + asin + "?pageNumber=2"
        response = urlopen(url)
        responseData = response.read()
        
        reviews = re.findall(r'<span data-hook="review-body" class="a-size-base review-text">(.*?)</span>', str(responseData))
        
        asinReviews=[]
        for review in reviews:
            reviewCleaned = re.sub('([0-9]{2}&#[0-9]{2}|<br />|\\\\x..|\\\\|&#([0-9]){2})','',review)
            asinReviews.append(reviewCleaned)

        reviewOutput[asin] = asinReviews
    
    #print(reviewOutput)
    
    dataFrame = []

    for asin in asins:
        reviewList = reviewOutput[asin]
        
        comprehendOutput = comprehendClient.batch_detect_sentiment(TextList=reviewList, LanguageCode='en')
        nlpResponseResultList = comprehendOutput['ResultList']

        
        for nlpResponse in nlpResponseResultList:
            score = nlpResponse['SentimentScore']
            
            recordType = [asin, 
                    score['Positive'], score['Negative'], score['Positive']-score['Negative'],
                    score['Neutral'], score['Mixed'], score['Neutral']-score['Mixed'],
                    nlpResponse['Sentiment']
                ]
            dataFrame.append(recordType)


    df = pd.DataFrame(dataFrame, columns=['ASIN','Pos','Neg', 'PosNeg', 'Neu', 'Mix', 'NeuMix', 'Sentiment'])
    asin0 = df.loc[(df["ASIN"] == asins[0])]
    asin1 = df.loc[(df["ASIN"] == asins[1])]

    bucketName = event['bucket']

    plt.figure()
    ax = plt.scatter(x=asin0["PosNeg"], y=asin0["NeuMix"], s=(1-asin0["Mix"])*1000, alpha=0.2, color='orchid', label=asins[0])
    ax = plt.scatter(x=asin1["PosNeg"], y=asin1["NeuMix"], s=(1-asin1["Mix"])*1000, alpha=0.2, color='orange', label=asins[1])
    plt.xlabel("Negavtive-to-Positive sentiment")
    plt.ylabel("Mixed-to-Neutral sentiment")
    plt.legend(loc=3,markerscale=0.5)
    plt.savefig('/tmp/scatter.png')
    s3Client.upload_file('/tmp/scatter.png', bucketName , 'scatter.png')

    plt.figure()
    sizes, labels, colors = pie_chart(asins[0], df)
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(asins[0])
    plt.savefig('/tmp/pie1.png')
    s3Client.upload_file('/tmp/pie1.png', bucketName, 'pie1.png')

    plt.figure()
    sizes, labels, colors = pie_chart(asins[1], df)
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(asins[1])
    plt.savefig('/tmp/pie2.png')
    s3Client.upload_file('/tmp/pie2.png', bucketName, 'pie2.png')

    return dataFrame

def pie_chart(asin, df):
    pos0 = df[(df["ASIN"]==asin) & (df["Sentiment"]=="POSITIVE")].count()[0]
    neg0 = df[(df["ASIN"]==asin) & (df["Sentiment"]=="NEGATIVE")].count()[0]
    mix0 = df[(df["ASIN"]==asin) & (df["Sentiment"]=="NEUTRAL")].count()[0]
    neu0 = df[(df["ASIN"]==asin) & (df["Sentiment"]=="MIXED")].count()[0]

    sizes = []
    labels = []
    colors = []
    if pos0 > 0:
        sizes.append(pos0)
        labels.append("Positive")
        colors.append("yellowgreen")
    if neu0 > 0:
        sizes.append(neu0)
        labels.append("Neutral")
        colors.append("lightskyblue")
    if mix0 > 0:
        sizes.append(mix0)
        labels.append("Mixed")
        colors.append("gold")
    if neg0 > 0:
        sizes.append(neg0)
        labels.append("Negative")
        colors.append("lightcoral")
    return sizes, labels, colors
