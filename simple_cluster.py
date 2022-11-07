from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='test',  remove=('headers', 'footers', 'quotes'))['data']

model = BERTopic()
topics, probabilities = model.fit_transform(docs)

model.get_topic(49)

model.visualize_distribution(probabilities[0])
