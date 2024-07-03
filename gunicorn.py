# gunicorn_config.py

import os
import sys

def remove_affixes(khmer_word, affixes):
    for affix in affixes:
        if khmer_word.startswith(affix):
            khmer_word = khmer_word[len(affix):]
        if khmer_word.endswith(affix):
            khmer_word = khmer_word[:-len(affix)]
    return khmer_word

def tokenize_kh(text, stopwords=khmer_stopword):
    # Tokenize the input text
    tokens_pos = khmernltk.pos_tag(text)
    
    # Filter out tokens with English characters, numerals, or punctuation, stopwords, and certain POS tags
    khmer_words = [
        remove_affixes(word, affixes) for word, pos in tokens_pos if (
            not re.search("[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]៖«»។៕!-…", word)
            and word not in stopwords
            and not any(char in word for char in '១២៣៤៥៦៧៨៩០')
            and pos not in ['o','.','1']
        )
    ]
    
    return khmer_words
