import samples
from elements import get_all_leaves, get_all_meta_content
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


def extract_text(node):
    return (e.xpath('./text()').extract_first()
            or e.xpath('./@content').extract_first())


ds = samples.get_dataset()

train = ds[:10]
test = ds[10:20]
validation = ds[20:]


page_elements_text = []

# testing only on first page for now
d = train[0]
page_elements = get_all_leaves(d['page']) + get_all_meta_content(d['page'])
page_elements_text = [extract_text(e) for e in page_elements]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(page_elements_text)
y = np.array([e == d['target'] for e in page_elements_text])

clf = svm.SVC()
clf.fit(X, y)

text_to_classify = [
    'elias dorneles',
    "Any mumbo jumbo here, it doesn't matter. :)",
]

# expecting: [True, False]
print(clf.predict(vectorizer.transform(text_to_classify)))
