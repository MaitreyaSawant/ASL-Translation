import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


with open('data_list.pickle','rb') as f:
    data = pickle.load(f)

with open('labels_list.pickle', 'rb') as f:
    labels = pickle.load(f)


lengths = [len(sample) for sample in data]
consistent_length = np.unique(lengths)
if len(consistent_length) != 1:
    print(f"Warning: Data lengths are inconsistent. Found lengths: {consistent_length}")
    consistent_length = consistent_length[0]  

    
    filtered_data = [sample for sample in data if len(sample) == consistent_length]
    filtered_labels = [label for i, label in enumerate(labels) if len(data[i]) == consistent_length]

    data = np.asarray(filtered_data)
    labels = np.asarray(filtered_labels)
else:
    data = np.asarray(data)
    labels = np.asarray(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)


score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
