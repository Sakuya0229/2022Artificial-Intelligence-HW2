from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import re

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)

    # Begin your code (Part 0)
    # function 1 remove chinese words
    preprocessed_text = re.sub('[\u4e00-\u9fa5]','',preprocessed_text)
    # function 2 remove Punctuation
    preprocessed_text = preprocessed_text.replace("<br / >","")
    preprocessed_text = re.sub('[!"#$%&\'()*+,-/-:;<=>@^_`{|}!~]','',preprocessed_text)
    # function 3 remove number
    preprocessed_text = re.sub('[0-9]','',preprocessed_text)
    # remove multi space and start end space
    preprocessed_text=re.sub(r"^\s+|\s+$", "", preprocessed_text)
    preprocessed_text = re.sub('[ ]+',' ',preprocessed_text)
    # End your code

    return preprocessed_text
