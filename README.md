## Usage of Daisi

It is recommended to use this application on the daisi platform itself using the link https://app.daisi.io/daisies/vijay/NLTK_for_Tamil/app. However, you can still use your own editor using the below method:

### First, load the Packages:

```
import pydaisi as pyd
nltk_for_tamil = pyd.Daisi("vijay/NLTK_for_Tamil")
```

### Now, connect to Daisi and access the functions using the following functions:

Join Words: To get the conjucted word

```
nltk_for_tamil.joinWords(word_a, word_b).value
```

Stemming: To find the root of the word

```
nltk_for_tamil.stemmer(token).value
```

Translation from Tamil to English:

```
nltk_for_tamil.translate(text).value
```

Splitting Content to Sentences:

```
nltk_for_tamil.split_content_to_sentences(content).value
```

Splitting Content to Paragraphs:

```
nltk_for_tamil.split_content_to_paragraphs(content).value
```

Sentiment Analysis: Finding whether the comment is OFFENSIVE or NOT-OFFENSIVE

```
nltk_for_tamil.senti(text).value
```

Date and Time Conversion:

```
nltk_for_tamil.date_n_time(a, b, c, d, e).value
```

Text Summarizer:

```
nltk_for_tamil.get_summary(title, content).value
```

## And done! We can now perform NLTK Operation even in Tamil easily!


