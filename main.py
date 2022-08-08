from __future__ import print_function
import pickle
from numpy import empty
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import tamil
import re
from translate import Translator
from tamilstemmer import TamilStemmer
from tamil.utf8 import splitMeiUyir, joinMeiUyir, get_letters, uyir_letters, mei_letters
from langdetect import detect
from datetime import datetime
from tamil.datetime import datetime

all_rules = {
    u'அ': {
        # todo : need to fine tune this solo words
        'first_solo_words': (u'என்ன', u'நல்ல', u'இன்ன', u'இன்றைய', u'படித்த', u'எழுதாத'),
        'secondword_first_chars': (u"க்", u"ச்", u"த்", u"ப்"),
        'diff_jn_words': (((u'சின்ன'), (u'சிறு', u'சிறிய'), u'ஞ்'), ),
        'special_secondword_first_chars': (uyir_letters, u'வ்'),
        'same_words': ((u'பல', u'சில'), u'ற்'),
        'same_word_disappear_lastchar': (u'அ'),
        'firstword_double_special_secondword': ((u'என்று', u'என'), u'வ்'),
    },
}

def joinWords(word_a, word_b):  
    word_a = word_a.strip()
    word_b = word_b.strip()
    # get readable letters of first word
    print(word_a)
    first_word_letters = get_letters(word_a)
    if first_word_letters[-1] in mei_letters:
        # first word last char is mei letter. so just return as it is.
        # todo : apply special conditions also
        rval = word_a + ' ' + word_b
        return rval
    # end of if first_word_last_chars[-1] in mei_letters:

    # get mei & uyir characters of first word's last char
    first_word_last_chars = splitMeiUyir(first_word_letters[-1])
    if len(first_word_last_chars) == 2:
        first_word_last_mei_char, first_word_last_uyir_char = first_word_last_chars
    else:
        first_word_last_mei_char, first_word_last_uyir_char = first_word_last_chars[0], first_word_last_chars[0]

    # get rule sub dictionary from all dictionary by passing
    rule = all_rules[first_word_last_uyir_char]

    if word_a == word_b:
        # both input words are same
        same_word_rule = rule.get('same_words', [])
        if word_a in same_word_rule[0]:
            # get conjuction char
            jn = same_word_rule[1]
            # insert conjuction char between input words
            rval = first_word_letters[0] + jn + word_b
            return rval
        elif len(first_word_letters) == 3:
            # both words are same but length is 3.
            disappear_lastchar = rule.get('same_word_disappear_lastchar', [])
            if disappear_lastchar:
                disappear_lastchar = disappear_lastchar[0]
                if first_word_last_uyir_char == disappear_lastchar:
                    first_word_first_char = first_word_letters[0]
                    # get uyir char of second word's first char
                    first_word_first_uyir_char = splitMeiUyir(first_word_first_char)[-1]
                    # get conjuction char by joining first word's last mei char and second word's first uyir char
                    jn = joinMeiUyir(first_word_last_mei_char, first_word_first_uyir_char)
                    # get first word till pre-last char
                    first_word = u''.join(first_word_letters[:-1])
                    # get second word from second char till end
                    second_word = u''.join(first_word_letters[1:])
                    # join all first, conjuction, second word
                    rval = first_word + jn + second_word
                    return rval
            # end of if disappear_lastchar:
        # end of if word_a in same_word_rule[0]:
    # end of if word_a == word_b:

    if word_a in rule.get('first_solo_words', []):
        # todo : need to find tune this first solo word check like using startswith, endswith, etc
        rval = word_a + ' ' + word_b
        return rval
    # end of if word_a in rule.get('first_solo_words', []):

    for diff_jn in rule.get('diff_jn_words', []):
        if word_a in diff_jn[0]:
            for last in diff_jn[1]:
                if word_b.startswith(last):
                    # apply different conjuction char rule
                    rval = word_a + diff_jn[2] + word_b
                    return rval
    # end of for diff_jn in  rule.get('diff_jn_words', []):

    # get readable letters of second word
    second_word_letters = get_letters(word_b)
    # get second word's from second char to till end
    second_word_after_first_char = u''.join(second_word_letters[1:])
    # get mei & uyir characters of second word's first char
    second_word_first_chars = splitMeiUyir(second_word_letters[0])
    if len(second_word_first_chars) == 2:
        second_word_first_mei_char, second_word_first_uyir_char = second_word_first_chars
    else:
        second_word_first_mei_char, second_word_first_uyir_char = second_word_first_chars[0], second_word_first_chars[0]

    if second_word_first_mei_char in rule.get('secondword_first_chars', []):
        # apply major conjuction rule
        return word_a + second_word_first_mei_char + ' ' + word_b
    # end of if second_word_first_mei_char in rule.get('secondword_first_chars', []):

    firstword_double_special_secondword = rule.get('firstword_double_special_secondword', [])
    if firstword_double_special_secondword:
        if len(first_word_letters) == 4:
            # check either first word has repeated two times
            if first_word_letters[:2] == first_word_letters[2:]:  # first word repeat two times within it
                # get root second word by removing prefix
                sec_word = second_word_first_uyir_char + second_word_after_first_char
                if sec_word in firstword_double_special_secondword[0]:
                    # get conjuction char by joining  special conjuction and  second root word
                    jn = joinMeiUyir(firstword_double_special_secondword[1], second_word_first_uyir_char)
                    # join all
                    return word_a + jn + second_word_after_first_char
    # end of if firstword_double_special_secondword:

    special_secondword_first_chars = rule.get('special_secondword_first_chars', [])
    if special_secondword_first_chars:
        if second_word_first_uyir_char in special_secondword_first_chars[0]:
            # get special conjuction char
            jn = special_secondword_first_chars[1]
            # join special conjuction char with second word's first uyir char
            second_word_first_schar = joinMeiUyir(jn, second_word_first_uyir_char)
            # complete second word with prefix of conjuction
            second_word = second_word_first_schar + second_word_after_first_char
            # join all
            return word_a + second_word
        # end of if second_word_first_uyir_char in special_secondword_first_chars[0]:
    # end of if special_secondword_first_chars:

    # if all above rules not applicable, then just return as it is !
    return word_a + ' ' + word_b

def stemmer(token):
    stemmer = TamilStemmer()
    token = stemmer.stemWord(token)
    return token

def translate(text):
    translator= Translator(from_lang="ta",to_lang="spanish")
    translation = translator.translate(text)
    return translation

def split_content_to_sentences(content):
    content = content.replace("\n", ". ")
    return content.split(". ")

# Naive method for splitting a text into paragraphs
def split_content_to_paragraphs(content):
    return content.split("\n\n")

# Caculate the intersection between 2 sentences
def sentences_intersection(sent1, sent2):
    
    # split the sentence into words/tokens
    # s1 = set(sent1.split(" "))
    # s2 = set(sent2.split(" "))
    s1 = set(tamil.utf8.get_letters(sent1))
    s2 = set(tamil.utf8.get_letters(sent2))
    
    # If there is not intersection, just return 0
    # if (len(s1) + len(s2)) == 0:
    if len(s1.intersection(s2)) == 0:
        return 0

    # We normalize the result by the average number of words
    return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2.0)

# Format a sentence - remove all non-alphbetic chars from the sentence
# We'll use the formatted sentence as a key in our sentences dictionary
def format_sentence(sentence):
    # sentence = re.sub(r'\W+', '', sentence)       # [\u0B80-\u0BFF]
    sentence = re.sub(r'\s+', '', sentence)
    sentence = re.sub(r'\d+','',sentence)
    # print sentence
    return sentence

# Convert the content into a dictionary <K, V>
# k = The formatted sentence
# V = The rank of the sentence
def get_sentences_ranks(content):

    # Split the content into sentences
    sentences = split_content_to_sentences(content)

    # Calculate the intersection of every two sentences
    n = len(sentences)
    values = [[0 for x in range(n)] for x in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            # Metric for intersection is symmetric so we calculate 1/2 only
            # For additional metrics see: ngram.Distance module in open-tamil
            # Ref https://github.com/Ezhil-Language-Foundation/open-tamil/blob/master/ngram/Distance.py
            if i >= j :
                values[i][j] = values[j][i]
                continue
            values[i][j] = sentences_intersection(sentences[i], sentences[j])

    # Build the sentences dictionary
    # The score of a sentences is the sum of all its intersection
    sentences_dic = {}
    for i in range(0, n):
        score = 0
        for j in range(0, n):
            if i == j:
                continue
            score += values[i][j]
        kw = format_sentence(sentences[i])
        if len(kw) != 0:
            sentences_dic[kw] = score
    
    return sentences_dic

# Return the best sentence in a paragraph
def get_best_sentence(paragraph, sentences_dic):

    # Split the paragraph into sentences
    sentences = split_content_to_sentences(paragraph)

    # Ignore short paragraphs
    if len(sentences) < 2:
        return ""

    # Get the best sentence according to the sentences dictionary
    best_sentence = ""
    max_value = 0
    for s in sentences:
        strip_s = format_sentence(s)
        if strip_s:
            if sentences_dic[strip_s] > max_value:
                max_value = sentences_dic[strip_s]
                best_sentence = s

    return best_sentence

# Build the summary
def get_summary(title,content):

    # Split the content into paragraphs
    paragraphs = split_content_to_paragraphs(content)
    sentences_dic = get_sentences_ranks(content)
    # Add the satle
    summary = []
    summary.append(title.strip())
    #summary.append("")

    # Add the best sentence from each paragraph
    for p in paragraphs:
        sentence = get_best_sentence(p, sentences_dic).strip()
        if sentence:
            summary.append(sentence)

    return ("\n").join(summary)


def date_n_time(a,b,c,d,e):
    d = datetime(int(a),int(b),int(c),int(d),int(e))
    textu = d.strftime_ta("%A (%d %b %Y) %p %I:%M")
    return textu

def vectorizing(text):
    loaded_vec = pickle.load(open('vectorizerrr.pk', 'rb'))
    input = [text]
    X = loaded_vec.transform(input)
    X = X.todense()
    return X

def inverse_encoder(predicted):
    file = open("le.obj",'rb')
    le_loaded = pickle.load(file)
    file.close()
    abc = le_loaded.inverse_transform(predicted)
    return abc[0]

def senti(text):
    densed = vectorizing(text)
    loaded_model = pickle.load(open('Tamil_senti.sav', 'rb'))
    UwU = loaded_model.predict(densed)
    Result = inverse_encoder(UwU)
    if Result == 'not':
        return 'Not offensive'
    else:
        return 'Offensive'



st.set_page_config(layout = "wide")

st.title("தமிழ் NLP")
st.write("##")

body1 = st.container()
body2 = st.container()
body3 = st.container()
body4 = st.container()

with body1: 
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.markdown("### Spell checker")
        st.write("Type in a word to check if the spelling is correct")
        word_spell = st.text_input("Enter a word")
        spellright = "It is a non-tamil word"
        langu = ""
        if word_spell:
            langu = detect(word_spell)
            if langu == "ta":
                st.markdown("Correct")
            else:
                inc = "Incorrect"
                st.markdown(f"<p style='text-align: center; color: black;'>{inc}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: black;'>{spellright}</h1>", unsafe_allow_html=True)

    with col2: 
        st.markdown("### Join words")
        st.write("Type in two words and get the conjucted word")
        word_join1 = st.text_input("Enter first word")
        word_join2 = st.text_input("Enter second word")
        joinedword = ""
        if word_join1 and word_join2:
            joinedword = joinWords(word_join1, word_join2)
        #join both the words
        st.markdown(f"<p style='text-align: center; color: black;'>{joinedword}</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown("### Date and time conventer")
        st.write("Convert date and time to tamil edho")
        date = st.date_input("Enter date")
        time = st.time_input("Enter time")
        tamildate = ""
        if date and time:
            date_split = date.strftime('%Y/%m/%d')
            time_split = time.strftime('%H:%M:%S')
            d = date_split.split("/")    #idhu list
            t = time_split.split(":")   #idhuvum list dhan
            tamildate = date_n_time(d[0],d[1],d[2],t[0],t[1])
            st.markdown(f"<p style='text-align: center; color: black;'>{tamildate}</h1>", unsafe_allow_html=True)

    st.write("---")

with body2:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Word stemmer")
        st.write("To stem a word (i.e to find the root of the word)")
        word_stem = st.text_input("Enter a word", key ='stem')
        df = ""
        if word_stem:
            df = stemmer(word_stem)
        st.markdown(f"<p style='text-align: center; color: black;'>{df}</h1>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Sentiment analysis")
        st.write("Finding whether the comment is OFFENSIVE or NOT-OFFENSIVE")
        word_senti = st.text_input("Enter a word", key ='sentiment')
        sf = ""
        if word_senti:
            sf = senti(word_senti)
        st.markdown(f"<p style='text-align: center; color: black;'>{sf}</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown("### Tamil to English translator")
        st.write("Enter a tamil word to translate it to english")
        word_translate = st.text_input("Enter a word", key ='translate')
        df = ""
        if word_translate:
            df = translate(word_translate)
        st.markdown(f"<p style='text-align: center; color: black;'>{df}</h1>", unsafe_allow_html=True)

    st.write("---")

with body3:
    st.markdown("### Text summarizer")
    st.write("Type in a paragraph, a summary of the paragraph is generated") 
    para_title = st.text_input("Enter title")
    paragraph = st.text_area(label = "Enter paragraph")
    textout = ""
    if para_title and paragraph:
        textout = get_summary(para_title, paragraph)
        st.markdown("Summarizered text :")
        st.write(textout)
    st.write("---")

with body4:
    st.markdown("### Grammar and spell check")
    st.write("Upload a file to check the spelling and grammar. If not, a new file is generated with the correct spelling and grammar") 
    input_file = st.file_uploader("Choose a file", type = ["txt"])
    if input_file:
        st.download_button("Download the corrected file", input_file)
