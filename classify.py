from __future__ import print_function
import numpy as np
from sklearn import cross_validation, svm, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import samples
from elements import get_all_leaves, get_all_meta_content, get_parent


def extract_text(node):
    text = (node.xpath('./text()').extract_first()
            or node.xpath('./@content').extract_first())
    if text is not None:
        return text.strip()


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
        parent_text = get_parent(sel).extract()
        text_around = parent_text.split(sel.extract())
        return text_around[0], text_around[-1]

    get_text_around = get_text_around_fast

    def add_count_features(feat_dict, prefix, tokens, use_position=False):
        for pos, tok in enumerate(tokens, 1):
            if use_position:
                key = prefix + str(pos) + '_' + tok
            else:
                key = prefix + '_' + tok
            feat_dict[key] = feat_dict.get(key, 0) + 1

    def word_class(s):
        if s.title() == s:
            return 'TT'
        if s.upper() == s:
            return 'UU'
        return 'O'

    def featurize(x):
        text = extract_text(x)
        tokens = tokenize(text)
        before, after = get_text_around(x)
        element_tag = x.root.tag
        parent = get_parent(x)

        features = {
            'element_tag': element_tag,
            'token_count': len(tokens),
        }

        if element_tag == 'meta':
            name = x.xpath('./@name').extract_first()
            if name:
                features['META_NAME'] = name

            meta_prop = x.xpath('./@property').extract_first()
            if meta_prop:
                features['META_PROP'] = meta_prop

        # in case current element or parent is a link, look into href tokens
        href = x.xpath('./@href') or parent.xpath('./@href')
        if href and href.extract_first():
            add_count_features(features, 'A_HREF_TOK',
                               tokenize(href.extract_first()))

        itemprop = x.xpath('./@itemprop')
        if itemprop:
            features['ITEMPROP'] = itemprop.extract_first()

        # look at element and parent classes equally
        classes = x.xpath('./@class').extract_first()
        if classes:
            add_count_features(features, 'CLASS', tokenize(classes))

        classes = parent.xpath('./@class').extract_first()
        if classes:
            add_count_features(features, 'CLASS', tokenize(classes))

        # look into tokens before and after, considering closest 5 tokens
        add_count_features(features, 'BEFORE', tokenize(before[-5:]))
        add_count_features(features, 'AFTER', tokenize(after[:5]))

        # ad-hoc word classes
        add_count_features(features, 'WORDC',
                           [word_class(w) for w in tokenize(text)[:10]],
                           use_position=True)

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


def main():
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
    X_train, y_train = X[:-20], y[:-20]
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

    import pickle
    with open('classifier.pickle', 'w') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()
