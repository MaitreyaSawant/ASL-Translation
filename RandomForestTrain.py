import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle files
with open('data_list.pickle','rb') as f:
    data = pickle.load(f)

with open('labels_list.pickle', 'rb') as f:
    labels = pickle.load(f)

# Check lengths of data samples
lengths = [len(sample) for sample in data]
consistent_length = np.unique(lengths)
if len(consistent_length) != 1:
    print(f"Warning: Data lengths are inconsistent. Found lengths: {consistent_length}")
    consistent_length = consistent_length[0]  # Use the first consistent length

    # Filter out inconsistent samples
    filtered_data = [sample for sample in data if len(sample) == consistent_length]
    filtered_labels = [label for i, label in enumerate(labels) if len(data[i]) == consistent_length]

    data = np.asarray(filtered_data)
    labels = np.asarray(filtered_labels)
else:
    data = np.asarray(data)
    labels = np.asarray(labels)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Evaluate the model
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
