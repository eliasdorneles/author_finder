import samples
from elements import get_all_leaves, get_all_meta_content
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import cross_validation


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
    return X, y


def maybe_truncate(s, size=75):
    return s if len(s) <= size else s[:size - 3] + '...'


ds = samples.get_dataset()

train = ds[:20]
test = ds[20:25]
validation = ds[25:]


text_X_train, y_train = build_Xy_from_pages_dataset(train)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(text_X_train)

clf = svm.LinearSVC()
clf.fit(X_train, y_train)


print('Robot trained!')
print('Now, trying to predict some hardcoded tests...')
text_to_classify = [
    'elias dorneles',
    'elias',
    'dorneles elias',
    "Any mumbo jumbo here, it doesn't matter. :)",
]

prediction = clf.predict(vectorizer.transform(text_to_classify))
expected = [True, True, True, False]
for t, p, e in zip(text_to_classify, prediction, expected):
    print('Worked? %5s  Got %5s, expected %5s (%r)' % (p == e, p, e, maybe_truncate(t)))


print('\nNow trying something from unseen data...')
text_X_test, y_test = build_Xy_from_pages_dataset(test[1:2])

print('Score', clf.score(vectorizer.transform(text_X_test), y_test))
