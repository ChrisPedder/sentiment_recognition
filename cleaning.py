"""
Routines to clean the text coming from the dataset, defines a pipeline which
can easily be extended to extra processing functions.
Routines:

* lowercase text
* remove formatting (\n etc)
* remove urls
* collapse alphabet to latin chars
* collapse mutliple whitespaces to single whitespace
* strip space at beginning and end

"""

import re

class DataProcessorPipeline:

    def __init__(self, processors=[]):
        """
        Initialize object with a list of text cleaning functions which will be
        applied one after the other.
        """
        self.processors = processors

    def process(self, text):
        for processor in self.processors:
            text = processor.process(text)
        return text


class Processor:

    def __init__(self, processing_function):
        self.processing_function = processing_function

    def process(self, text):
        return self.processing_function(text)

def lower(text):
    return text.lower()

format_re = re.compile('[\t\n\f\r]')
def remove_formatting(text):
    return format_re.sub(' ', text)

url_re = re.compile('\\w+://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
                    '(?:%[0-9a-fA-F][0-9a-fA-F]))+')
def remove_urls(text):
    return url_re.sub('', text)

latin_alpha = re.compile('[^a-zA-Z0-9 ]')
def force_alphabet(text):
    return latin_alpha.sub(' ', text)

multispace = re.compile('[ ]{2,}')
def remove_whitespace(text):
    return multispace.sub(' ', text)

def strip(text):
    return text.strip()

def remove_stop_words(text):
    text_tokens = word_tokenize(text)

def split_words(text):
    return text.split(' ')

# Returns a tokenized list of words in the original sentence
LogisticPipeline = DataProcessorPipeline([
    Processor(lower),
    Processor(remove_urls),
    Processor(remove_formatting),
    Processor(force_alphabet),
    Processor(remove_whitespace),
    Processor(strip),
    Processor(split_words)
])


TransformerPipeline = DataProcessorPipeline([
    Processor(remove_urls),
    Processor(remove_formatting),
    Processor(remove_whitespace),
    Processor(strip)
])

if __name__ == '__main__':
    text = 'Just been to my local supermarket and all that was left was: \r\r\n2 pre-made sandwiches.\r\r\n1 pack of Mr. Kipling French fancies and one bottle of dandelion and burdock.\r\r\nLuckily I was picnic buying!\r\r\n#Covid_19\r\r\n#StopPanicBuying'
    print(text)
    print(LogisticPipeline.process(text))
