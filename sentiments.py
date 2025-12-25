# Doing sentiment alanlysis on the dataset with LLM model without streamlit model integration.
# Doing cluster analysis as well

import pandas as pd
import time 
from transformers import pipeline
import json
# loading and reading dataset 

data=pd.read_csv("chat_conversations_dataset.csv")

print(data.head())

# dataset has conversation_id,product , its description, customer_ratings,num_turns,full conservation

# checking for missing values and removing them thus performing EDA

print(data.isnull().sum())


# Since no missing values we will now move to do anlaysis with LLM Model's
# using different model and using them to compare the results
model1="distilbert-base-uncased-finetuned-sst-2-english" # its free model

#comparing resuts of different llm models for sentiment analysis aling with time taken by these models
    
analysis1=pipeline("sentiment-analysis",model=model1)

def analyse_chats(full_conversation):
    
    t1_start=time.time()
    result1=analysis1(full_conversation)
    t1_final=time.time()-t1_start
    actual_rating = data.loc[data["full_conversation"] == full_conversation,"customer_rating"].values[0]
    
    return {
        
        
        "Model result":result1,
        "Model time":float(t1_final),
        "score_given by Model":int(actual_rating)
        
    }
    
result=data['full_conversation'].apply(analyse_chats)

print(json.dumps(result.to_list(),indent=4,ensure_ascii=False))







    
     
    
    
    
    




    
    
    
    
    
    
    








    
    
    
    
    

    

    
    

