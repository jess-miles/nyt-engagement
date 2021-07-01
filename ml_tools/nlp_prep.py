import html
import re
import string
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def clean_docs(doc):
    """
    Performs a few basic cleaning steps:
    - Unescapes any escaped HTML (i.e. `&amp;` for ampersand)
    - Replaces URLs with the `{link}` placeholder text
    - Replaces non-ASCII characters with spaces
    - Replaces any control characters with spaces
    - Finally, cleans up any instances of multiple spaces that might have
    been created during this process, replacing with a single space
    
    Takes in a document at a time, and returns the cleaned document text as 
    a string.
    """
    # unescape HTML characters
    doc = html.unescape(doc)
    
    # remove URLs and links, replacing them with existing placeholder
    urls = re.findall("http[^ ]+|www\.[^ ]+", doc)
    for url in urls:
        doc = str.replace(doc, url, '{link}')
    
    # replace non-ASCII characters with space
    doc = re.sub(r"[^\x00-\x7F]+", ' ', doc)
    
    # replace ASCII control characters with space
    doc = re.sub(r"[\x00-\x1F]", ' ', doc)
    
    # remove multiple spaces, which will exist after all this replacing words
    doc = re.sub(r"[ ]{2,}", ' ', doc)
    
    return doc

def get_pattern_hits(doc, pattern, out_type):
    """Takes in a regex pattern and a string of text, and checks for the 
    presence of the pattern in the string. 
    
    Returns a copy of the text string with the pattern replaced with spaces, 
    and the matches. See `out_type` for available formats for the match output.
    
    *** Arguments
    
    doc: string. Text to be searched for the pattern.
    
    pattern: string, regex pattern. Pattern to be searched for in the string.
    
    out_type: string. Indicate how / whether the matches on the pattern
    should be returned.
    
        If `out_type` = `list`, each match on the pattern is saved into a list
        and the list is returned. 
        
        If `out_type` = `string`, each match on the pattern is saved into a 
        string separated by spaces and that string is returned. 

        If `out_type` = `boolean`, the output is just
        True if there was at least one match, and false if there were none.

        If `out_type` = `none`, the pattern matches are simply removed from the
        text string and not logged in any way.
    """
    
    # determine the variable type for recording hits
    if out_type=='list': 
        hits = []
    elif out_type=='string': 
        hits = ""
    elif out_type=='bool':
        hits = False
    elif out_type=='none':
        hits = None
        
    # search for regex pattern in doc
    pattern_hits = re.findall(pattern, doc)

    if len(pattern_hits) > 0:
        # replace the hits in the original doc string
        # need to use the re version otherwise substrings won't be replaced 
        # correctly!
        doc = re.sub(pattern, ' ', doc)

        # replace multiple spaces with a single space
        doc = re.sub(r"(\s{2,})", ' ', doc)

        # Update appropriate hits variable
        for hit in pattern_hits:
            if out_type=='list':
                hits.append(hit)
            elif out_type=='string':
                hits = hits + ' ' + hit
            elif out_type=='bool':
                hits = True

        if out_type=='list':
            hits = list(set(hits))
                 
    return doc, hits

def pattern_match_in_df(df, doc_col, hit_col, pattern, out_type='list', 
                        replace=True):
    """Loops through values in a particular dataframe columns, and searches
    for regex pattern matches. 
    
    Returns the same dataframe, updated with new `hit_col`, and replaced
    `doc_col`, depending on arguments passed.
    
    *** Arguments
    
    df: Dataframe containing the `doc_col` to be searched.
    
    doc_col: String. Name of the dataframe column containing the text to be
    searched for the pattern.
    
    hit_col: String. Name of the column which should be added to the dataframe
    when it's returned, containing the pattern hit information.
    
    pattern: string, regex pattern. Pattern to be searched for in the string.
    
    out_type: string. See `get_pattern_hits` documentation for details.
    
    replace: Boolean, default True.
    
    Use `replace=True` to have the matches be replaced with spaces, and the
    original `doc_col` replaced with a new one with matches removed.
    If `replace=False`, the `doc_col` will not be updated, and matches will
    only be logged in the new `hit_col`.
    
    """
    updates = []
    
    # loop through each row in the dataframe to process its record
    for i in df.index:
        if df.at[i, doc_col] is not np.nan:
            new_doc, hits = get_pattern_hits(df.at[i, doc_col], pattern, out_type)
            updates.append([i, new_doc, hits])
        else:
            updates.append([i, df.at[i, doc_col], None])
    
    # create a dataframe out of the updated info
    # use original dataframe index so don't have to reset the index before
    # applying this
    df_new = pd.DataFrame(updates, columns=['index', doc_col, hit_col])
    df_new.set_index('index', inplace=True)

    # if we were told not to output the hits, drop the hits column
    if out_type=='none':
        df_new.drop(columns=[hit_col], inplace=True)
    
    # if we were told not to replace the hits in the original text column,
    # drop the new doc column
    if not replace:
        df_new.drop(columns=[doc_col], inplace=True)
    
    if len(df_new.columns) > 0:
        df = df.join(df_new, lsuffix='_old', how='inner')
        if replace:
            df.drop(columns=[f"{doc_col}_old"], inplace=True)
    else:
        print("Dataframe was returned as-is based on arguments passed.")
            
    return df

def generate_wordcloud(docs, cmap, stopwords, min_font_size=14, n_grams=True, 
                       title='Word cloud'):
    """Generate a wordcloud from a list of pre-tokenized words. Words will be
    joined into a space-delimited string inside the function.
    """
    cloud = WordCloud(colormap=cmap, width=600, height=400, 
                      prefer_horizontal=0.95, min_font_size=min_font_size,
                     collocations=n_grams)\
                      .generate_from_text(" ".join(docs))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Display the generated image:
    ax.imshow(cloud)
    ax.set_axis_off()
    ax.set_title=(title)
    ax.margins(x=0, y=0);

def generate_freqs_wordcloud(df, word_col, freq_col, cmap, min_font_size=14,
                            title='Word cloud'):
    """Generate a wordcloud from a dataframe where one column holds the words
    and another column holds the frequency or weight. The dictionary for the
    wordcloud is generated inside the function.
    """
    freq_dict = pd.Series(df[freq_col].values,index=df[word_col]).to_dict()
    
    cloud = WordCloud(colormap=cmap, width=600, height=400, 
                      prefer_horizontal=0.95, min_font_size=min_font_size,
                     collocations=False).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Display the generated image:
    ax.imshow(cloud)
    ax.set_axis_off()
    ax.set_title=(title)
    ax.margins(x=0, y=0);


def plot_wordfreqs(df, word_col, freq_col, top_n, sub_title):
    """Displays a horizontal bar plot of words frequencies from the `top_n`
    words in a corpus.
    
    `df` should be a DataFrame where `word_col` is the column name containing 
    the word or ngram text, and `freq_col` is the column name containing the 
    frequency.
    """
    
    with sns.plotting_context(context='talk'):
        fig, ax = plt.subplots(figsize=(8, top_n / 2.5))
        sns.barplot(data=df,
                    y=df[word_col][:top_n], 
                    x=df[freq_col][:top_n], color='blue')
        ax.set_title(f"Top {top_n} Words\n{sub_title}")


def tokenize_corpus_dict_tweet(df, target_vals, stop_list=None, 
                               verbose=True, target_col='emotion',
                              doc_col='cleaned'):
    """ Tokenizes text and separates the tokens. Returns a dictionary where the
    keys are target class labels and the values are a list of tokens within 
    the labeled class.
    
    `df` is a Dataframe with the document text and labels, and `doc_col` is the
    name of the column in which document text is stored.
    
    `target_vals` should be a list of the class labels; labels in `target_col` 
    but not in the `target_vals` list will be ignored.
    
    If `stop_list` (type=list) provided, those words or tokens will be removed
    from the returned dict.
    """
    tweettokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

    # generate corpus for each emotion
    corpus_per_target = {}

    for val in target_vals:
        
        if verbose:
            print(f"Starting target val: {val}")

        # get series of text docs per target_val
        docs = df.loc[df[target_col]==val, doc_col]

        # loop through docs and tokenize each one
        corpus = []

        i = 0
        for doc in docs:
            # tokenize using tweet tokenizer
            tokens = tweettokenizer.tokenize(doc)
            
            # remove stop words if needed
            if not stop_list == None:
                tokens = [token for token in tokens if token not in stop_list]
            
            # remove words if they're just spaces!
            tokens.remove(' ') if ' ' in tokens else None
            
            corpus.extend(tokens)
            i += 1
            
            if verbose and (i % 1000 == 0):
                print(f"Processed {i} docs out of {len(docs)}...")

        # add corpus to dict
        corpus_per_target[val] = corpus
        
    if verbose:
        print(f"Done!")
    return corpus_per_target


def tokenize_lemma_stem(doc):
    """
    Applies TweetTokenization, then lemmatization and stemming in one shot.
    
    Uses NLTK, so assumes the appropriate NLTK classes have been imported.
    
    Uses TweetTokenizer to tokenize documents first, and remove handles.
    Then uses NLTK lemmatization on each token.
    Finally, applies stemming to each token.
    """
    tweettokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # tokenize using TweetTokenizer
    tokens = tweettokenizer.tokenize(doc)
    
    # lemmatize using NLTK
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # stem using NLTK
    tokens = [stemmer.stem(token, ) for token in tokens]
    
    return tokens