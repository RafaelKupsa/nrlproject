# nrlproject
Programming project for the seminar Neural Representation Learning, LMU München SS21

## Motivation

This project was realized for the seminar "Algorithmische und formale Aspekte II" (Neural representation learning) at the LMU university of Munich in the summer semester 2021.
The goal of the project was to become familiar with working with transformer architectures and in the process comparing two different models (BERT and CANINE) on their performance in next sentence classification in poetry as well as rhyming sentence classification.

## Code

There are two python files contained in this project's repository.
Data.py deals with data extraction and preprocessing. The data is extracted from the Chicago Rhyming Poetry Corpus which can be found at https://github.com/sravanareddy/rhymedata and consists of raw data files with poetry from various authors seperated into verses and annotated with author, titles and the rhyme structure of each verse. During extraction all files in the english_raw folder containing poetry (the ".txt" files) are considered.

There are two Data classes that were coded for this project. The first is called NextLineData which extracts equal amounts of true and false examples for next sentence prediction from the corpus, meaning sentence pairs consisting of a verse line and the next verse line for the true examples and sentence pairs consisting of a verse line and a random line from anywhere in the corpus for the false examples.

The second is called RhymingData which extracts equal amounts of true and false examples for predicting rhyming sentences from the corpus. For that the annotated rhyme structure is taken into account and so true examples consist of sentence pairs in the same verse that rhyme and false examples of sentence pairs in the same verse that don't rhyme. The idea behind this was to see if a pre-trained model could be fine-tuned to classify rhyming sentences correctly disregarding any semantic relations, so to achieve equal semantic relations in the true and the false sentence pairs, both were only chosen to span one verse maximally.


Model.py contains the main script as well as the code for loading and fine-tuning the models. The main script automatically fine-tunes both a BERT and a CANINE model and saves it together with a file containing stats about the training process. Both models are loaded and preprocessed using the transformers python module by HuggingFace (https://huggingface.co/).

The main difference between the two models is that in Bert the input sequences are separated into tokens on the word level which have to be mapped to predefined ids (Words that are not contained in the id dictionary are split according to the WordPiece algorithm first.) while in Canine the sequences are separated into tokens on the character level meaning every character's id is its unicode representation. Hyperparameters are the same for both models. Originally I intended to use a larger batch size but it soon became apparent that my GPU could not handle more than one training example at a time which is why the batch size is consistently 1. For the sake of keeping the training time short I did not use all of the extracted training examples but experimented with 3 epochs with 10000 training examples each as well as only 1 epoch with 30000 training examples. As will be explained in the results, more training examples would probably not have had a lot of influence on the outcome.

The optimizer used was AdamW, also from the transformers module, with a learning rate of 2e-5.

## Results

As stated in the motivation, the project was intended for me to become familiar with applied transfer learning which was mainly achieved. The other purpose was to compare two different models on the mentioned tasks. My hypothesis was the following: Since the CANINE model uses input on character level it would fare better in dealing with poetry than BERT because poetry relies on subword-level, structural components of words more than prose, such as syllable count and rhymes.

At first I tried to investigate this using classical next sentence prediction (extracting data with the NextLineData class): Giving the models sentence pairs immediately next to each other in a verse with a positive classification and random sentence pairs with a negative classification. However this resulted in the BERT model performing much better than the CANINE model. As documented in the file "next_line_poetry_10000ex_3epochs.stats" the BERT model's best accuracy was 84% (with the accuracy decreasing each episode) and the CANINE model's 50% (which is as good as a random guess). Admittedly both of these results could suggest flaws in my implentation, so should definitely be further investigated but due to lack of time, were left at that.

A problem with the next line format would be that the structural components of poetry often span the whole verse and are often lost in two consecutive lines. Rhyming lines often occur in alternating or more complex patterns (e.g. A B A B, A A B C C B) instead of being directly next to each other, as do verse metrics. The models may therefore still rely mainly on semantic similarities between words to judge whether two lines are likely to occur in sequence. This could explain why BERT still outperforms CANINE here.

Taking this into account I wanted to try to more explicitly provide the models with examples which would be structurally similar. Therefore I extracted data with the RhymingData class (as explained above) and trained both models feeding them examples of definitely rhyming sentences with a positive classification and definitely not rhyming sentences with a negative classification while keeping the sentence pairs contained to one verse to avoid fine-tuning on the semantic similarities. This turned out to not work very well, as can be seen in the files "rhyming_poetry_10000ex_3epochs.stats" and "rhyming_poetry_30000ex_1epoch.stats" which in all cases had accuracies close to 50% meaning they performed not better than chance.

So my takeaway from the project is that transformers trained on natural language tasks such as next sentence prediction and language modelling cannot easily be fine-tuned to deal with structural similarities between sentences which in hindsight should maybe be obvious.

## Prerequisites

Python version 3.8

Python libraries:
- numpy
- torch
- transformers
- built-in: os, time, random

## Features

This project contains the following files and directories:
- Data.py
- Model.py
- next_line_poetry_10000ex_3epochs.stats
- rhyming_poetry_10000ex_3epochs.stats
- rhyming_poetry_30000ex_1epoch.stats

The files for the fine-tuned models are unfortunately too large to upload here.
