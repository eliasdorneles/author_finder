import samples
from elements import get_all_leaves, get_all_meta_content
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


def extract_text(node):
    return (node.xpath('./text()').extract_first()
            or node.xpath('./@content').extract_first())


def get_page_elements(page):
    return get_all_leaves(page) + get_all_meta_content(page)


def extract_elements_text(elements):
    return [extract_text(e) for e in elements]


ds = samples.get_dataset()

train = ds[:20]
test = ds[20:25]
validation = ds[25:]


target = []
input_text = []

for d in train:
    elements_text = extract_elements_text(get_page_elements(d['page']))

    input_text.extend(elements_text)
    target.extend([d['target'] == e for e in elements_text])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(input_text)
y = np.array(target)

clf = svm.LinearSVC()
clf.fit(X, y)


def maybe_truncate(s, size=75):
    return s if len(s) <= size else s[:size - 3] + '...'


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
for d in test[1:2]:
    print('For article: %s' % d['url'])
    elements_text = extract_elements_text(get_page_elements(d['page']))
    prediction = clf.predict(vectorizer.transform(elements_text))
    expected = [d['target'] == e for e in elements_text]

    for t, p, e in zip(elements_text, prediction, expected):
        print('Worked? %5s  Got %5s, expected %5s (%r)' % (p == e, p, e, maybe_truncate(t)))
