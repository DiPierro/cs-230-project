"""lda.py"""
# Utility functions
import csv
import pdb
from collections import OrderedDict
import joblib
import random
from operator import itemgetter
from numpy import geomspace
import argparse
import os
import sys
import logging

# Gensim functions for topic modeling
from gensim.models import Phrases
# from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import gensim.downloader

# Utility functions provided in CS230 project template
from model.utils import Params, set_logger

# Global settings - necessary because csv is so large
csv.field_size_limit(sys.maxsize)
PYTHON = sys.executable

# Initialize argparse parser
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/num_topics',
                    help="Directory containing params.json")
parser.add_argument('--train_path', default='data/processed/train/train_200.csv',
                    help="File containing train data")
parser.add_argument('--hyperparameter', default='num_topics',
                    help="Hyperparameter to tune")
parser.add_argument('--dev_path', default='data/processed/dev/dev_200.csv',
                    help="File containing dev data")
parser.add_argument('--test_mode', default=False,
                    help="Run script in test mode")

def process_data(path, params, dictionary=None):
    """Prepare data for LDA model and filter out most uncommon/common words"""
    # Define place names as stopwords
    stop_words = ["alameda", "burlingame", "cupertino", "hayward", "hercules", "mountain", "view", "mtc", "oakland", "san", "francisco", "jose", "leandro", "mateo", "santa", "clara", "stockton", "sunnyvale"]
    
    # Read in the data
    logging.info("Reading in the data...")
    data = path
    docs = []
    titles = []

    with open(data, 'r') as f:
        csv_text = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
        csv.field_size_limit(sys.maxsize)
        
        counter = 0
        for row in csv_text:
            # Skip the header row
            if row != ['doc_id', 'place', 'date', 'doc_title', 'doc_type', 'doc_text']:
                tokens = []
                # Reformat the incoming text
                text = row[-1][2:-2].replace("'", "").split(",")
                title = '{}-{}'.format(row[0], counter)
                for token in text:
                    token = token.strip()
                    if token != ' ' and token not in stop_words:
                        tokens.append(token)
                docs.append(tokens)
                titles.append(title)
    logging.info("Done reading in the data.")
    
    # Add bigrams
    bigram = Phrases(docs, min_count = 30)
    for i in range(len(docs)):
        for token in bigram[docs[i]]:
            if '_' in token:
                docs[i].append(token)

    if dictionary is None:
        # Create a dictionary representation of the documents
        # if there is not already one
        logging.info("Creating dictionary...")
        dictionary = Dictionary(docs[1:])
        # We treat no_below and no_above as hyperparameters to tune.
        dictionary.filter_extremes(no_below=params.no_below, no_above=params.no_above)
        logging.info("Done creating dictionary.")

    # Create a bag-of-words representation of the documents.
    logging.info("Creating the corpus...")
    corpus_long = [dictionary.doc2bow(doc) for doc in docs]
    # Remove empty lists created because of filter_extremes
    corpus = list(filter(None, corpus_long))
    logging.info("Done creating the corpus.")
    
    return corpus, dictionary, titles

def train_lda(corpus, params, dictionary):
    """Train LDA model according to provided params"""
    # Set training parameters.
    num_topics = params.num_topics
    chunksize = params.chunksize
    passes = params.passes
    iterations = params.iterations
    decay = params.decay
    offset = params.offset  

    # Make an index to word dictionary.
    logging.info("Mapping ids to words...")
    temp = dictionary[0]
    id2word = dictionary.id2token
    logging.info("Done mapping ids to words.")

    logging.info("Making the LDA model...")
    lda = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=3, # Allows algorithm to run more efficiently
        chunksize=chunksize,
        alpha='asymmetric', # If low: Each document is represented by only a few topics
        eta='auto', # If low: Each topic is only represented by a few words
        decay=decay,
        offset=offset,
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=None,
        random_state=230,
        per_word_topics=True
    )
    logging.info("Done making the LDA model.")
    return lda

def log_results(lda, params, corpus, dictionary, train_flag, args):
    """Print topics, calculate coherence and export results to csv."""
    if train_flag:
        logging.info("Logging training results...")
    else:
        logging.info("Logging dev set results...")
    # Log top words from each topic and topic coherence scores 
    top_topics = lda.top_topics(corpus)
    lda.print_topics(num_topics=-1, num_words=20)
    if train_flag:
        logging.info("Finished logging training results...")
    else:
        logging.info("Finished logging dev results...")

    logging.info("Evaluating average topic coherence...")
    # Compute average topic coherence
    # We want topic coherence as close to 0 as possible
    avg_topic_coherence = sum([t[1] for t in top_topics]) / params.num_topics

    # Calculate UMass coherence as a sanity check
    # This should be identical to avg_topic_coherence
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    umass_coherence = cm.get_coherence() 

    # Calculate GloVe coherence score
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='c_w2v', keyed_vectors=glove_vectors)
    glove_coherence = cm.get_coherence()

    # Put this data into a dictionary to save
    metadata_dict = OrderedDict([
            ('avg_umass_coherence', avg_topic_coherence),
            ('umass_coherence', umass_coherence),
            ('glove_coherence', glove_coherence),
            ('num_topics', params.num_topics),
            ('chunksize', params.chunksize),
            ('passes', params.passes),
            ('no_above_list', params.no_above),
            ('no_below_list', params.no_below),
            ('decay', params.decay),
            ('offset', params.offset)
        ])

    # Check if csv exists
    if train_flag:
        csv_path = os.path.join(args.parent_dir, 'train_results_200.csv')
    elif not train_flag and not args.test_mode:
        csv_path = os.path.join(args.parent_dir, 'dev_results_200.csv')
    elif not train_flag and args.test_mode:
        csv_path = os.path.join(args.parent_dir, 'test_results.csv')
    
    new_file = not os.path.exists(csv_path)

    # Write out our results
    with open(csv_path, 'a') as out_file:       
        dict_writer = csv.DictWriter(out_file, metadata_dict.keys())
        if new_file:
            dict_writer.writeheader()
        dict_writer.writerow(metadata_dict)
    logging.info("Average topic coherence saved.")
    
    # Save the model
    if train_flag:
        logging.info("Saving model...")
        model_name = 'num_topics_{}_chunksize_{}_passes_{}_no_above_{}_no_below_{}_decay_{}_offset_{}.jl'.format(
            params.num_topics,
            params.chunksize,
            params.passes,
            params.no_above,
            params.no_below,
            params.decay,
            params.offset
            )
        joblib.dump(lda, os.path.join(args.parent_dir, model_name))
        logging.info("Model saved.")

def eval_results(dev_path, lda, dictionary, params):
    """Calculate coherence on unseen documents and print topics for unseen documents."""
    logging.info("Evaluating results...")
    corpus, _, titles = process_data(path=dev_path, params=params, dictionary=dictionary)
    # Get the topic proportions for each document in dev
    log_results(lda, params, corpus, dictionary, False, args)
    
    # Get main topic in each new document
    # Adapted from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
    for i, row_list in enumerate(lda[corpus]):

        row = row_list[0]        
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the 3 most-dominant topics for each document
        # Also get the weight of the topic in the document
        logging.info("Original doc: " + titles[i])
        for j, (topic_num, prop_topic) in enumerate(row):
            if j < 2:
                words = lda.show_topic(topicid=topic_num, topn=30)
                topic_keywords = ", ".join([word for word, prop in words])
                logging.info("Topic number: " + str(topic_num))
                logging.info("Topic proportion: " + str(round(prop_topic,4)))
                logging.info("Topic keywords: " + str(topic_keywords))
                
    logging.info("Finished evaluating results on dev set.")
    
def run_search(search_param, args, params):
    """Train a model and evaluate the model for a set of parameters during parameter search"""
    # Define unique job name
    job_name = "{}_{}".format(args.hyperparameter, search_param)

    # Set the logger
    set_logger(os.path.join(args.parent_dir, job_name + '.log'))

    # Train the model
    corpus, dictionary, _ = process_data(path=args.train_path, params=params, dictionary=None)
    lda = train_lda(corpus, params, dictionary)

    # Save results
    log_results(lda, params, corpus, dictionary, True, args)

    if not args.test_mode:
        # Evaluate the model on dev set
        eval_results(dev_path=args.dev_path, lda=lda, dictionary=dictionary, params=params)
    else:
        # Evaluate model in the test set
        eval_results(dev_path='data/processed/test/test_200.csv', lda=lda, dictionary=dictionary, params=params)

if __name__ == "__main__":
    random.seed(230)
    
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform search over one parameter at a time
    # Edit these lists to see results for different values
    num_topics_list = [100] # Number of topics
    chunksizes = [256, 1024, 4096, 16384] # How many documents to process at a time
    passes = geomspace(start=1, stop=500, num=10, dtype='int16') # Number of epochs
    no_above_list = geomspace(start=0.5, stop=1, num=10)# Filter out words that occur in more than X/total documents
    no_below_list = [1, 5, 10, 15, 20, 25] # Filter out words that occur in less than X documents
    decay = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    offset = [384, 512, 640, 768, 896]
    
    if args.hyperparameter == 'num_topics':
        search_params = num_topics_list
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.num_topics = search_param
            run_search(search_param, args, params)
    
    elif args.hyperparameter == 'chunksize':
        search_params = chunksizes
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.chunksize = search_param 
            run_search(search_param, args, params)
    
    elif args.hyperparameter == 'no_above':
        search_params = no_above_list
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.no_above = search_param 
            run_search(search_param, args, params)

    elif args.hyperparameter == 'no_below':
        search_params = no_below_list
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.no_below = search_param 
            run_search(search_param, args, params)
            
    elif args.hyperparameter == 'passes':
        search_params = passes
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.passes = search_param 
            run_search(search_param, args, params)
    
    elif args.hyperparameter == 'decay':
        search_params = decay
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.decay = search_param 
            run_search(search_param, args, params)
    
    elif args.hyperparameter == 'offset':
        search_params = offset
        for search_param in search_params:
            # Modify the relevant parameter in params
            params.offset = search_param 
            run_search(search_param, args, params)