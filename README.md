# README

This repository contains code for the Project Milestone assignment for CS 230.

The first set of files in this directory are scripts and associated outputs excerpted and adapted from TopicVec, the source code for "Generative Topic Embedding: a Continuous Representation of Documents" (ACL 2016) by Shaohua Li, Tat-Seng Chua, Jun Zhu and Chunyan Miao.
The full implementation of TopicVec is here: https://github.com/askerlee/topicvec

In particular, the scripts `csv2topic.py`, `topicvecDir.py` and `utils.py` have been adapted to take as input a ~20,000-row csv containing 55-word chunks of agenda documents and minutes documents from [Agenda Watch](www.agendawatch.org)'s corpus of local government meeting documents. They have also been edited to correct syntax errors and encoding issues in the original code.

The document `55word_doc_text-10.28.8.log` contains the output log from an initial run of `csv2topic.py`. This log shows the top words in each of 20 topics formed over 100 EM iterations. Within each topic, the log reports the relevance and similarity of individual words.

The document `55word_doc_text-em100-best.topic.vec` reports the topic embedding produced by the model.


