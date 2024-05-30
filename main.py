import requests
from time import sleep
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


# get all links on the main page
resp = requests.get('https://edition.cnn.com/')

soup = BeautifulSoup(resp.text, "html.parser")
soup = soup.find(class_="layout__wrapper layout-homepage__wrapper")
links = soup.find_all('a')


# parce all the links
pages = []
targets = []

for _ in links:
    link = _['href']
    category = ''
    # print(link)

    if link[:8] != 'https://':
        category = link.split('/')
        category = category[4] if category[1].isdigit() and len(category[1]) == 4 else category[1]
        link = f"https://edition.cnn.com{link}"
    else:
        category = 'n/a'
        continue
    print(category)

    try:
        resp = requests.get(link)
        soup = BeautifulSoup(resp.text, "html.parser")
        all_text = soup.get_text(separator='\n', strip=True)

        pages.append(all_text)
        targets.append(category)

        sleep(0.5)
    except requests.exceptions.RequestException as e:
        print(f"smth went wrong for {link}: {e}")


# prepare the model
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(pages)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(targets)

model = MultinomialNB()  # or LogisticRegression()
model.fit(x, y)


# make prediction:
predict_urls = [
    "https://www.bbc.com/news/articles/c100mjqrm6zo",
    "https://www.bbc.com/news/articles/c1994g22ve9o",
]

all_text = []

for url in predict_urls:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator='\n', strip=True)
    all_text.append(text)

X_text = vectorizer.transform(all_text)

predicted_category = model.predict(X_text)
predicted_category_name = label_encoder.inverse_transform(predicted_category)
print(f"Most likely categories: {predicted_category_name})")