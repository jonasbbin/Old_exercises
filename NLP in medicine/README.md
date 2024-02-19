# Disclaimer:
This was an old exercise. Therefore it might be possible that current approaches largely differ from this approach. Also keep in mind that this task was solved under potential restrictions (solution approaches, model usage, hardware, time).
# NLP in medicine
The idea of this project is to get hands on experience when working with texts. In this exercise we are working with the PubMed 200k RCT dataset. The datasets consists of PubMed abstracts. The aim is to classify sententeces into different categories: background, objective, method, result and conclustion. This is useful to facilitate literature reviewing.

This exercise consists of 3 tasks:
- Preprocess the dataset (lowercasing, stop-words removal..), obtain TF-IDF scores and train classifier to predict abstract class
- Train a word embedding model like Word2Vec, obtain sentence embeddings and train classifier to predict abstract class
- Evaluate the performance of BERT on the same taks. Use a pre-trained model with no fine-tuning and a pre-trained model with fine-tuning
