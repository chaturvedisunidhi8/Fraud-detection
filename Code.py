import pandas as pd   
import numpy as np     
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout  

# Load Dataset   
df = pd.read_csv("info.csv")     
print(df.shape)  
print(df.head())  

# Basic Cleaning   
df.dropna(inplace=True)  
df.reset_index(drop=True, inplace=True)   

# Input and output features  
X = df['text']         
y = df['label']         

# Tokenization   
max_words = 5000  # vocabulary size  
max_len = 100  
  
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")  
tokenizer.fit_on_texts(X)  
  
X_seq = tokenizer.texts_to_sequences(X)  
X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')  
  
# Train-Test Split   
X_train, X_test, y_train, y_test = train_test_split(  
    X_pad, y, test_size=0.33, random_state=42  
)  
  
# Model   
model = Sequential([  
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),  
    LSTM(64, return_sequences=False)
    Dropout(0.5)
    Dense(1, activation='sigmoid')  
])  
  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
model.summary()  
  
# Train   
model.fit(  
    X_train, y_train,  
    validation_data=(X_test, y_test),  
    epochs=5,  
    batch_size=64  
)  
  
# Predict   
y_pred_probs = model.predict(X_test)  
y_pred = np.where(y_pred_probs > 0.5, 1, 0)  
  
# Evaluation   
print(accuracy_score(y_test, y_pred))  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
