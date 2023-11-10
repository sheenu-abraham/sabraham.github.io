---
title: "Binary Prediction of Smoker Status using Bio-Signals"
tags:
- R
- blogdown
date: "2023-11-05"
---

This starter code will help you with this Kaggle  https://www.kaggle.com/competitions/playground-series-s3e24

```python

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
train=pd.read_csv('./train.csv')
train.head()
```

```python
train.describe()
```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = train.copy()
y = X.pop('smoking')


X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

```


```python
from tensorflow import keras
from tensorflow.keras import layers

# YOUR CODE HERE: define the model given in the diagram
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(units=512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
  
    layers.Dense(units=256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
  
    layers.Dense(units=256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
  
    layers.Dense(units=256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
  
    layers.Dense(units=256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1,activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
```



```python
# Predict on test
test=pd.read_csv('./test.csv')
test.head()
```



```python
test_df = preprocessor.transform(test)
```


```python
from tensorflow.keras.models import Sequential, save_model, load_model
# Save the model
filepath = './saved_model'
save_model(model, filepath)
```

```python
predictions = model.predict(test_df)

```
```python
sub=test[['id']]
df = pd.DataFrame(predictions, columns=['smoking'])
submission=pd.concat([sub, df], axis=1)
submission.to_csv('submission.csv',index=False)
```


```python
prediction_classes = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]

```

This will get you a score of 0.86174 in the leaderboard. 
