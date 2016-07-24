from __future__ import print_function
import numpy as np
from sklearn import cross_validation, svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

import samples
from elements import get_all_leaves, get_all_meta_content


def extract_text(node):
    return (node.xpath('./text()').extract_first()
            or node.xpath('./@content').extract_first())


def get_page_elements(page):
    return get_all_leaves(page) + get_all_meta_content(page)


def extract_elements_text(elements):
    return [extract_text(e) for e in elements]


def build_Xy_from_pages_dataset(dataset):
    X = []
    y = []
    for d in dataset:
        elements_text = extract_elements_text(get_page_elements(d['page']))
        y.extend([d['target'] == e for e in elements_text])
        X.extend(elements_text)
    return X, np.array(y)


def create_classifier():
    return make_pipeline(
        CountVectorizer(),
        svm.LinearSVC(),
    )


def get_trained_classifier(X_train, y_train):
    """Return classifier trained with given dataset parameters
    """
    clf = create_classifier()
    clf.fit(X_train, y_train)
    return clf


dataset = samples.get_dataset()

X, y = build_Xy_from_pages_dataset(dataset)
clf = create_classifier()

# this gives the prediction result for every element
# when it was in the test dataset during cross validation
predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)

cm = metrics.confusion_matrix(y, predicted)
print('\nConfusion matrix:')
print(cm, '\n\n')
print(metrics.classification_report(y, predicted))

# does this make sense to measure too?
# scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring='f1')
# print("Accuracy: %0.2f (+/- %.05f)" % (scores.mean(), scores.std() * 2))


print('Training and peeking at the word weights...')
X_train, y_train = build_Xy_from_pages_dataset(dataset[:20])
clf = get_trained_classifier(X_train, y_train)
cv = clf.steps[0][1]
svc = clf.steps[1][1]
word_weights = zip(svc.coef_[0], cv.vocabulary_)

print('Top 10 weights for negative cases')
for weight, word in sorted(word_weights)[:10]:
    print('%0.5f  %s' % (weight, word))

print('\nTop 10 weights for positive cases')
for weight, word in sorted(word_weights)[-10:][::-1]:
    print('%0.5f  %s' % (weight, word))
