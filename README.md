# ZUM_NLP_Project


# Sentiment Analysis Project
## Dataset Overview
### Source
This dataset is sourced from Kaggle, specifically designed for sentiment analysis tasks.

# Context
'''Training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative.''' - Dataset description from Kaggle

# Content
The dataset comprises tweets formatted in a CSV file with emoticons removed. It includes the following fields:

- Polarity: Indicates the sentiment of the tweet (0 = negative, 2 = neutral, 4 = positive).
- Tweet ID: The unique identifier of the tweet.
- Date: The date when the tweet was posted.
- Query: The query used (if any). 'NO_QUERY' indicates no specific query.
- User: The username of the tweet's author.
- Text: The content of the tweet.

## Model Description
### Model_CNN_embedding_A
This model is designed for sentiment analysis using Convolutional Neural Networks (CNN) with embedded layers. Achieved 0.73 F1 Score for the task and test set accuracy of 72.59%.

### Key Components
Embedding Layer: Uses GloVe embeddings to transform text data into fixed-size dense vectors.
Convolutional Layers: Two sets of Conv1D layers for feature extraction, each followed by MaxPooling1D for dimensionality reduction.
Batch Normalization: Applied after the first Conv1D layer for model stability and efficiency.
Dropout Layer: Added to prevent overfitting.
Flatten Layer: Converts pooled feature maps into a single column that is passed to the fully connected layer.
Dense Layers: Two dense layers for classification, with the last layer using softmax activation for multiclass classification.
Model Construction
The model is built using Keras' Sequential API, specifying the architecture in a linear stack of layers.

### Training and Evaluation
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. It is trained on tokenized training data and evaluated on a test set.

### Usage
To use this model:

Prepare the Data: Tokenize your text data and ensure it matches the input format of the model.
Load GLoVe Embeddings and create embedding matrix
Load the Model: Instantiate Model_CNN_embedding_A with the appropriate vocabulary size and max length parameters.
Compile the Model: Use the compile method with specified optimizer and loss function.
Train the Model: Fit the model to your data using the fit method.
Evaluate the Model: Assess model performance on unseen data using the evaluate method.
