#This module will provide freqency of common_issues faced by customers using clustering     
# using TF-IDF vectorizer and KMeans clustering to identify common issues from customer complaints

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import json
# loading dataset
data=pd.read_csv("chat_conversations_dataset.csv")
# dataset has conversation_id,product , its description, customer_ratings,num_turns,full conservation
# extracting issue descriptions from full conversations
issue_descriptions=data['full_conversation'].tolist()
# vectorizing the issue descriptions using TF-IDF
vectorizer=TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(issue_descriptions)
# applying KMeans clustering to identify common issues
num_clusters=5
kmeans=KMeans(n_clusters=num_clusters,random_state=42)
kmeans.fit(X)
# assigning cluster labels to issue descriptions
data['cluster_label']=kmeans.labels_
# getting the frequency of each cluster to identify common issues
common_issues_frequency=data['cluster_label'].value_counts().to_dict()
print("Common Issues Frequency:")
print(json.dumps(common_issues_frequency,indent=4))
# displaying common issues with their descriptions
for cluster_num in range(num_clusters):
    print(f"\nCommon Issue Cluster {cluster_num}:")
    cluster_issues=data[data['cluster_label']==cluster_num]['full_conversation'].tolist()
    for issue in cluster_issues[:5]:  # displaying first 5 issues in each cluster
        print(f"- {issue}")


# The output will show the frequency of common issues and sample issue descriptions from each cluster.


