from __future__ import print_function
import numpy as np
import re
from sklearn import cross_validation, svm, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

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
    page_labels = []
    for i, d in enumerate(dataset):
        elements = get_page_elements(d['page'])
        for e in elements:
            text = extract_text(e)
            X.append(e)
            y.append(d['target'] == text)
            page_labels.append(i)
    return X, np.array(y), np.array(page_labels)


def extract_dict_features(X):
    def tokenize(x):
        return re.split('\s', x)

    def featurize(x):
        text = extract_text(x)
        tokens = tokenize(text)
        parent_text = x.xpath('./parent::*').extract_first()
        text_around = parent_text.split(x.extract())
        before_tokens = tokenize(text_around[0])
        after_tokens = tokenize(text_around[-1])
        parent_tokens = set(tokenize(parent_text))
        return {
            'elem_tag': x.xpath('name()').extract_first(),
            'token_count': len(tokens),
            'before_tokens': len(before_tokens),
            'after_tokens': len(after_tokens),
            'by': 'by' in before_tokens,
            'written': 'written' in parent_tokens,
            'wrote': 'wrote' in parent_tokens,
            'posted': 'posted' in parent_tokens,
            'submitted': 'submitted' in parent_tokens,
        }

    return [featurize(x) for x in X]


def create_classifier():
    return make_pipeline(
        FunctionTransformer(extract_dict_features, validate=False),
        DictVectorizer(),
        svm.LinearSVC(),
    )


def get_trained_classifier(X_train, y_train):
    "Return classifier trained with given dataset parameters"
    clf = create_classifier()
    clf.fit(X_train, y_train)
    return clf


dataset = samples.get_dataset()

X, y, page_labels = build_Xy_from_pages_dataset(dataset)
clf = create_classifier()

# this gives the prediction result for every element
# when it was in the test dataset during cross validation
cv_iter = cross_validation.LabelKFold(page_labels, n_folds=10)
predicted = cross_validation.cross_val_predict(clf, X, y, cv=cv_iter)

cm = metrics.confusion_matrix(y, predicted)
print('\nConfusion matrix:')
print(cm, '\n\n')
print(metrics.classification_report(y, predicted))


print('Training and peeking at the word weights...')
X_train, y_train, _ = build_Xy_from_pages_dataset(dataset[:20])
clf = get_trained_classifier(X_train, y_train)
cv = clf.steps[-2][1]
svc = clf.steps[-1][1]
word_weights = zip(svc.coef_[0], cv.vocabulary_)

print('Top 10 weights for negative cases')
for weight, word in sorted(word_weights)[:10]:
    print('%0.5f  %s' % (weight, word))

print('\nTop 10 weights for positive cases')
for weight, word in sorted(word_weights)[-10:][::-1]:
    print('%0.5f  %s' % (weight, word))
