from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Species')
test_y = test.pop('Species')

def input_fun(features, labels, training= True, batch_size= 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

my_feature_col = []
for key in train.keys():
    my_feature_col.append(tf.feature_column.numeric_column(key = key))
print(my_feature_col)

Classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_col,
    hidden_units=[30,10],
    n_classes=3
    )
Classifier.train(
    input_fn=lambda: input_fun(train, train_y, training=True ),
    steps=5000
)

evalute_result = Classifier.evaluate(
    input_fn=lambda: input_fun(train, train_y, training=False )
)

def input_fn(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type Numeric value as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ":")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]
predictions = Classifier.predict(input_fn= lambda: input_fn(predict) )
for pre_dict in predictions:
    class_id = pre_dict['class_ids'][0]
    probability = pre_dict['probabilities'][class_id]
    print(SPECIES[class_id], 100* probability)
