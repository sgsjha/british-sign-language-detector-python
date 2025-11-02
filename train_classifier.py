#train_classifier.py

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open('./data.pickle', 'rb'))

# --- FIX: make all feature vectors the same length before np.asarray ---
lengths = [len(v) for v in data_dict['data']]
target_len = max(set(lengths), key=lengths.count)  # use the most common length

def _pad_or_trim(v, L):
    v = np.asarray(v, dtype=float).ravel()
    if len(v) >= L:
        return v[:L]
    return np.pad(v, (0, L - len(v)), mode='constant')

data = np.vstack([_pad_or_trim(v, target_len) for v in data_dict['data']])
labels = np.asarray(data_dict['labels'])
# --- end FIX ---

#split the information into 2 diff sets of data
#test = 20% of the data, rest is used to train
#shuffle : shuffle the data (good practice), good for img classifications to remove biases
#stratify : split but keep the same proportions as train set in the test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of the samples were classified correctly!'.format(score*100))


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()







