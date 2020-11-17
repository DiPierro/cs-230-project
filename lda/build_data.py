"""build_data.py"""
# Libraries------------------

import difflib
import os
from pprint import pprint
import pdb
import re
from collections import Counter
import spacy
import en_core_web_sm
import csv
from operator import itemgetter 
import random

# Code-----------------------

def preprocess(file_list, places):
    """Processes all of the documents."""
    print("Preprocessing documents...")
    new_list = []
    
    for place in places:
        print("Processing ", place)
        baseline_flag = False
        baseline_checked = False
        counter = 0
        for doc in file_list:
            # Check if we have 'diffed' against a baseline document
            # This is to remove 'boilerplate' text where we can
            if doc['place'] == place and not baseline_flag:
                baseline = doc
                baseline_flag = True
            elif doc['place'] == place and baseline_flag:
                counter += 1
                if not baseline_checked:
                    baseline['processed_text'] = process_text(text = diff_doc(baseline=doc['raw_text'], new_doc=baseline['raw_text']))
                    assert 'doc_id' in baseline.keys()
                    new_list.append(baseline)
                    baseline_checked = True
                doc['processed_text'] = process_text(text = diff_doc(baseline=baseline['raw_text'], new_doc=doc['raw_text']))
                assert 'doc_id' in doc.keys()
                new_list.append(doc)
        
        # Add any document that can't be diffed
        if counter == 0:
            baseline['processed_text'] = process_text(text = ''.join(baseline['raw_text']))
            assert 'doc_id' in baseline.keys()
            new_list.append(baseline)

    print("Finished preprocessing documents.")

    return new_list

def read_files(directory):
    """Read in text files to be processed."""
    print("Reading files...")
    file_list = []
    places = []
    files = os.listdir(DIR)
    
    for index, item in enumerate(files):
        
        # Add docs to file_list
        file_dict = {}
        
        file_name = directory + "/" + files[index]
        file_dict['raw_text'] = open(file_name, 'r').read().splitlines()
        file_dict['doc_id'] = item
        file_dict['place'] = re.search(r"^.*(?=_20)", item).group(0)
        file_dict['date'] = re.search(r"20\d{2}-\d{2}-\d{2}(?=_)", item).group(0)
        file_dict['doc_title'] = re.search(r"(?<=\d{2}_).*(?=_\w+)", item).group(0)
        file_dict['doc_type'] = re.search(r"(Agenda|Minutes)(?=\.txt)", item).group(0)
        assert 'doc_id' in file_dict.keys()
        file_list.append(file_dict)
        # Add places to list of places
        if file_dict['place'] not in places:
            places.append(file_dict['place'])
    assert len(files) == len(file_list)
    print("Finished reading files.")
    return places, file_list
        
def diff_doc(baseline, new_doc):
    """Compare a given doc to a baseline doc to remove boilerplate text."""
    d = difflib.Differ()
    result = list(d.compare(baseline, new_doc))
    results = []
    for line in result:
        if line.startswith('+'):
            results.append(line[1:])
    return ' '.join(results)

def process_text(text):
    """Remove useless POS and stop words, lemmatize, lowercase."""
    # Setup

    sp_text = nlp(text)
    pos_list = ['AUX', 'ADP', 'CONJ', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'PUNCT', 'SYM', 'SPACE', 'X']

    # Part of speech filter
    pos_filter = []
    pos_tags = [(word, word.pos_) for word in sp_text]
    for tag in pos_tags:
        if tag[1] not in pos_list:
            pos_filter.append(str(tag[0]).lower())
    
    # Identify entities
    ents = set()
    for ent in sp_text.ents:
        if ent.label_ in ["PERSON"]:
            ents.add(str(ent).lower())
    
    # Remove stopwords
    default_stops = nlp.Defaults.stop_words
    default_stops |= {
        'agenda','minutes','council','meeting','planning','mayor','ordinance','district',
        'special','county','city','municipal','commission','committee',
        'amendment','district','https','notice','event','staff','comment' 
        'participate','report','doc_text','report','advisory','name','team',
        'printed', 'public', 'public hearing', 'office', 'manager', 'supervisor', 
        'board', 'clerk', 'motion', 'consent', 'calendar', 'finance', 'printed'
        'record', 'code', 'section', 'attorney', 'staff', 'report', 'item', 
        'clerk', 'chair', 'page', 'meet', 'plan', 'call', 'participate', 'ada',
        'civil', 'remotely', 'consider', 'library', 'budget', 'commissions'
    }
    default_stops |= ents
    
    default_stops |= add_stop_words('data/processed/top_words.csv')
    
    lower = [word.lower() for word in pos_filter]
    no_stops = [word for word in lower if word not in default_stops]

    no_short = [word for word in no_stops if len(word) != 1]
    
    return no_short

def add_stop_words(word_file):
    """Reads csv file of 200 most-common words into list."""
    csv_top_words = open(word_file)
    top_words = csv.reader(csv_top_words)
    top_words_set = set()
    for word in top_words:
        top_words_set.add(word[0])
    return top_words_set

def get_top_words(file_list):
    """
    Counts the most common words in our corpus. This function was used to generate top_words.csv.
    It is not currently invoked in main() since top_words.csv is provided.
    """
    word_dict = {}
    print("Getting top words...")
    for doc in file_list:
        for word in doc['diff_text']:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] = word_dict[word] + 1
    top_200 = dict(sorted(word_dict.items(), key = itemgetter(1), reverse = True)[:200])
    print("Finished getting top words.")

    with open('data/processed/top_words.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in top_200.items():
            writer.writerow([key])

def chunk(processed_files, num_words):
    """Splits each document into chunks of between len(num_words) to len(num_words*1.5)"""
    new_list = []
    for doc in processed_files:
        assert ('doc_id' and 'processed_text') in doc.keys()
        full_text = doc['processed_text']
        for i in range(0, len(full_text), num_words):
            # Check how many words will be left in the doc after this chunk
            if len(full_text[i:i + num_words*2]) > (num_words*1.5):
                new_dict = {}
                new_dict['doc_id'] = doc['doc_id']
                new_dict['place'] = doc['place']
                new_dict['date'] = doc['date']
                new_dict['doc_title'] = doc['doc_title']
                new_dict['doc_type'] = doc['doc_type']
                new_dict['doc_text'] = full_text[i:i + num_words]
                assert ('doc_id' and 'doc_text') in new_dict.keys()
                new_list.append(new_dict)
            # If there will be very few words left in the doc
            # then combine the next chunk with the current chunk
            else:
                new_dict = {}
                new_dict['doc_id'] = doc['doc_id']
                new_dict['place'] = doc['place']
                new_dict['date'] = doc['date']
                new_dict['doc_title'] = doc['doc_title']
                new_dict['doc_type'] = doc['doc_type']
                new_dict['doc_text'] = full_text[i:i + num_words*2]
                assert ('doc_id' and 'doc_text') in new_dict.keys()
                new_list.append(new_dict)
    print("Finished chunking text.")
    return new_list

def split(final_files):
    """Split data into train/dev/test sets."""
    assert 'doc_id' in final_files[0].keys()
    final_files = sorted(final_files, key = lambda i: i['doc_id'])
    random.seed(230)
    random.shuffle(final_files)
    split_1 = int(0.8 * len(final_files))
    split_2 = int(0.9 * len(final_files))
    train = final_files[:split_1]
    dev = final_files[split_1:split_2]
    test = final_files[split_2:]
    return train, dev, test

def write(data, name):
    """Helper method to write train/dev/test sets to csv files."""
    print("Writing ", name, "data...")
    file_name = 'data/processed/' + name + '/' + name + '_200.csv'
    headers = data[0].keys()
    with open(file_name, 'w') as csv_file:
        dict_writer = csv.DictWriter(csv_file, headers, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(data)

if __name__ == '__main__':
    
    DIR = "data/raw/content/docs"
    nlp = spacy.load('en_core_web_sm') 
    places, file_list = read_files(directory=DIR)
    processed_files = preprocess(file_list, places)
    assert len(file_list) == len(processed_files)
    final_files = chunk(processed_files, 200)
    train, dev, test = split(final_files)
    write(train, "train")
    write(dev, "dev")
    write(test, "test")
    

