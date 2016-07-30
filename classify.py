from __future__ import print_function
import numpy as np
from sklearn import cross_validation, svm, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
    tokenize = CountVectorizer().build_tokenizer()

    def get_text_around_slow(sel):
        """Slower, but more accurate way of getting text around"""
        from autopager import htmlutils
        return htmlutils.get_text_around_selector_list([sel])[0]

    def get_text_around_fast(sel):
        """Cheap attempt of getting text around, inaccurate for
        nested elements or for elements repeating inside parent"""
        parent_text = sel.xpath('./parent::*').extract_first()
        text_around = parent_text.split(sel.extract())
        return text_around[0], text_around[-1]

    get_text_around = get_text_around_fast

    def featurize(x):
        text = extract_text(x)
        tokens = tokenize(text)
        before, after = get_text_around(x)

        features = {
            'element_tag': x.root.tag,
            'token_count': len(tokens),
        }

        for tok in tokenize(before[-5:]):
            k = 'BEFORE_' + tok
            features[k] = features.get(k, 0) + 1

        for tok in tokenize(after[:5]):
            k = 'AFTER_' + tok
            features[k] = features.get(k, 0) + 1

        for pos, tok in enumerate(text.split()[:10], 1):
            k = 'WORD_%s_%s' % (pos, tok)
            features[k] = features.get(k, 0) + 1

        return features

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
