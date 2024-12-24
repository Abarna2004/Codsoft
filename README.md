# Codsoft_ML Projects

## 1) Movie Genre Classification

Technology Stack: Python, Scikit-learn, Pandas, Numpy, TF-IDF, Logistic Regression, Naive Bayes

In this project, you will build a machine learning model to classify movies into various genres based on their descriptions. The dataset contains movie titles, genres, ratings, and descriptions.

- **Dataset**: Contains movie details such as name, genre, rating, director, and actors.
- **Machine Learning Task**: Genre classification based on movie descriptions using text features extracted from the description.

### Steps:
1. **Load the dataset**: Import the dataset containing movie details (description and genre).
2. **Text preprocessing**: Clean the movie descriptions (tokenization, removing stop words).
3. **Feature extraction**: Convert the cleaned text into numerical features using techniques like TF-IDF.
4. **Model training**: Use algorithms like Logistic Regression or Naive Bayes to classify the genres.
5. **Evaluation**: Measure performance using metrics such as accuracy, precision, recall, and F1-score.

### Classification Output:
![image](https://github.com/user-attachments/assets/77df2ebd-c437-4367-9245-1044b78c93a8)

---

## 2) Spam Detection

Technology Stack: Python, Scikit-learn, Pandas, TF-IDF, Logistic Regression

This project focuses on classifying SMS messages as either spam or ham (legitimate). Using a labeled dataset of SMS messages, you will train a model that can predict whether a given message is spam.

- **Dataset**: The dataset consists of SMS messages labeled as "ham" (legitimate) or "spam."
- **Machine Learning Task**: Text classification for spam detection, using features extracted from the message content.
- **Model**: Logistic Regression is used for the classification task.

### Steps:
1. **Load the dataset**: Import the SMS dataset (containing message text and labels).
2. **Preprocessing**: Clean the text and vectorize using TF-IDF.
3. **Model training**: Train a Logistic Regression model to classify messages as "spam" or "ham."
4. **Evaluation**: Evaluate the model's performance using classification metrics like accuracy and F1-score.

### Output:
![image](https://github.com/user-attachments/assets/6bde9be2-8d69-4f62-88c0-2d3ed58ff70c)

---

## 3) Customer Churn Prediction

Technology Stack: Python, Pandas, Scikit-learn, Random Forest, StandardScaler

In this project, you will predict customer churn for a telecommunications company based on various customer attributes. The goal is to classify customers into two categories: those who are likely to churn and those who are likely to stay.

- **Dataset**: Contains customer details such as age, gender, credit score, balance, and account status.
- **Machine Learning Task**: Predict customer churn using classification algorithms such as Random Forest.

### Steps:
1. **Load the dataset**: Import the customer churn dataset (`Churn_Modelling.csv`).
2. **Data Preprocessing**: Drop irrelevant columns and encode categorical features using one-hot encoding.
3. **Feature scaling**: Normalize features using StandardScaler.
4. **Model Training**: Train a classification model (Random Forest) to predict churn.
5. **Evaluation**: Evaluate the model's performance using accuracy, confusion matrix, and classification report.

### Prediction:
![image](https://github.com/user-attachments/assets/7a87e6ac-c18e-47b6-a23b-780e68ea72fd)
