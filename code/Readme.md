# Machine Learning Algorithms
For this project, we will utilize two machine learning algorithms frequently used for the similar research purposes: BERT (Bidirectional Encoder Representations from Transformers) and Multi-Task Learning (MTL). The effectiveness of this MLAs in hypothesis testing comes from the initial definition: BERT processes text bi-directionally, considering both the left and right contexts of each word, allowing the model to capture complex dependencies and relationships within the text. Additionally, BERT is pre-trained on a massive amount of data, allowing it to capture general language features effectively, and its semantic understanding can be particularly helpful in distinguishing between factual information and rumors and objective or biased publication.


## Sample code for data query (generated with the help of ChatGPT 3.5):
```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load dataset
data = load_fake_news_dataset()

# Preprocess data
processed_data = preprocess_data(data)

# Split data into training and testing sets
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize and convert text to BERT embeddings
train_inputs = tokenizer(train_data['text'], padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(train_data['label'].values)

test_inputs = tokenizer(test_data['text'], padding=True, truncation=True, return_tensors='pt')
test_labels = torch.tensor(test_data['label'].values)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

# Evaluate the model
accuracy = accuracy_score(test_labels.numpy(), predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))
```

### Let me walk you through the steps in the data query process:
1. Data Collection: We will first search for datasets containing social media posts, news articles, forum discussions, or other user-generated content potentially containing misinformation posted within the social networks of interest. The current two datasets for the project contain three kinds of data (already preprocessed, that is the next step): fake, real and satirical.
2. Data Preprocessing: We will preprocess the collected text data from step one. This process utilizes methods, such as tokenization, removing stopwords, lemmatization, and encoding the text into a format suitable for BERT.
3. Fine-Tuning BERT: In order to make BERT particularly effective for misinformation detection, fine-tuning is required. This involves training the BERT model on a labeled dataset of fake, real and satire examples. Each piece of text is labeled as either "fake", "real", or "satire", and then given additional labels of "rumor", "fact", "biased", "objective". 
4. Classification: For enhanced performance of the previous step, we will utilize a classification model, which can be as simple as logistic regression or more complex neural network architectures. This model is trained to predict whether a given text contains fake information or not.
5. Evaluation: In this step, we evaluate the model's performance using metrics like precision, recall, F1-score, and accuracy.


## A pseudo-code for data query process (generated with the help of ChatGPT 3.5):
```python
# Importing necessary libraries
import BERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Loading BERT pre-trained model
bert_model = BERT.load_model()

# Loading dataset
data = load_fake_news_dataset()

# Preprocessing data
processed_data = preprocess_data(data)

# Spliting data into training and testing sets
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Tokenizing and converting text to BERT embeddings
train_features, train_labels = tokenize_and_embed(train_data, bert_model)
test_features, test_labels = tokenize_and_embed(test_data, bert_model)

# Training logistic regression classifier
classifier = LogisticRegression()
classifier.fit(train_features, train_labels)

# Making predictions on the test set
predictions = classifier.predict(test_features)

# Evaluating the model
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))
```
