import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import Word, TextBlob
import re

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
import pandas as pd

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have",
                       "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                       "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                       "he's": "he is",
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                       "I've": "I have",
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
                       "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                       "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not",
                       "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                       "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                       "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                       "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                       "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                       "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have",
                       "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have",
                       "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                       "why've": "why have",
                       "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have",
                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would",
                       "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would",
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are",
                       "you've": "you have"}
misspell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many',
                 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',
                 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                 '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'lucio': 'overwatchhero',
                 'zenyatta': 'overwatchhero', 'zarya': 'overwatchhero', 'pharah': 'overwatchhero',
                 'widowmaker': 'overwatchhero', 'torb': 'overwatchhero', 'torbjorn': 'overwatchhero',
                 'genji': 'overwatchhero',
                 'hanzo': 'overwatchhero', 'roadhog': 'overwatchhero', 'junkrat': 'overwatchhero',
                 'mei': 'overwatchhero',
                 'dva': 'overwatchhero', 'doomfist': 'overwatchhero', 'reinhardt': 'overwatchhero',
                 'sombra': 'overwatchhero',
                 'brigitte': 'overwatchhero', 'moira': 'overwatchhero', 'mccree': 'overwatchhero',
                 'orisa': 'overwatchhero',
                 'baptiste': 'overwatchhero', '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
                 '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth', '9th': 'ninth', '10th': 'tenth',
                 'akamaized': 'music', 'overwatchheros': 'overwatchhero', 'myanimelist': 'anime',
                 'symmetra': 'overwatchhero',
                 '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight',
                 '9': 'nine', '0': 'zero'}
misspell_dict1 = {'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth', '6th': 'sixth',
                  '7th': 'seventh', '8th': 'eighth', '9th': 'ninth'}


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)


def reduce_lengthening(txt):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", txt)


def correct_spelling(x, dic):
    if bool(x.isdigit()) or bool(x.isalpha()):
        for word in dic.keys():
            x = x.replace(word, dic[word])
    return x


def correct_spelling1(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


# import data
df = pd.read_csv('reddit_train.csv')

############################################################################################################

# add features
df['word_count'] = df['comments'].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df['comments'].str.len()
df['avg_word'] = df['comments'].apply(lambda x: avg_word(x))
df['num_stopwords'] = df['comments'].apply(lambda x: len([x for x in x.split() if x in stopwords]))
df['num_numbers'] = df['comments'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df['num_upper'] = df['comments'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

# lower case
df['comments'] = df['comments'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# remove contractions
df['comments'] = df['comments'].apply(lambda x: clean_contractions(x, contraction_mapping))
# remove possessive
df['comments'] = df['comments'].str.replace("'s", '')
# replace symbols with spaces
df['comments'] = df['comments'].str.replace('[^a-zA-Z0-9\s]', ' ')
# remove whitespaces
df['comments'] = df['comments'].str.replace('[^\w\s]', '')
# replace numbers with #
df['comments'] = df['comments'].apply(lambda x: " ".join(clean_numbers(x) for x in x.split()))
# correct spelling of frequently misspelled words
df['comments'] = df['comments'].apply(lambda x: " ".join(correct_spelling(x, misspell_dict) for x in x.split()))
df['comments'] = df['comments'].apply(lambda x: " ".join(correct_spelling1(x, misspell_dict1) for x in x.split()))
# fix word lengthening
df['comments'] = df['comments'].apply(reduce_lengthening)
# correct spelling
# df['comments'] = df['comments'].apply(lambda x: str(TextBlob(x).correct()))
# lemmatize and remove stopwords
nrows = len(df)
lemmatized_text_list = []
for row in range(0, nrows):
    lemmatized_list = []  # Create empty list containing lemmatized words
    text = df.loc[row]['comments']  # Save the text and its words into an object
    text_words = text.split(" ")
    for word in text_words:  # Iterate through every word to lemmatize
        if word not in stop_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)  # Join the list
    lemmatized_text_list.append(lemmatized_text)  # Append to the list containing the texts
df['comments'] = lemmatized_text_list

# output to csv
df.to_csv('./preprocessed_data2.csv', encoding='utf-8', index=False)
