# Machine Learning Project


# Enhancing Tagalog-to-English Machine Translation with BART

My project fine-tunes a pre-trained BART model to improve Tagalog-to-English machine translation, with a focus on pronoun accuracy. 

## Requirements
The following libraries/packages needed for this project (cn also be imported in the jupyter notebook itself):
- pandas
- transformers
- torch
- sacrebleu
- scikit-learn
- jupyterlab

Dataset Format:
The dataset (train1.csv) contains the following columns:
Tagalog Sentences (example below):
Siya ay nagluto. 
Tagged Sentence: Tagalog sentences with annotations for pronouns(example below):
Siya [singular, third-person, gender-neutral] ay nagluluto.
English Translation: English translations of the Tagalog sentences (example below):
They are cooking.
Pronoun Annotation: The pronoun annotations (English) for each sentence. (example below):
Pronoun = "they," singular, gender-neutral
There are a total of 604 Tagalog-English sentence pairs and the overall count of sentences (including annotation and tagged is 2416)

### Running the jupyter notebook:
Run the notebook titled MLProj
Run each cell in order.

Notebook Structure:
Data loading and preprocessing:
Loads train1.csv.
Tokenizes Tagged Sentence as (input) and English Translation as (target). Note that you can specify Tagalog if you want to train untagged Tagalog sentences as input. 

Model training:
Fine-tunes facebook/bart-large. (BART) 
Uses train_test_split to divide the dataset (80% training, 20% validation).
Early stopping based on validation loss included.

Evaluation:
Calculates BLEU, Precision, Recall, F1 Score, and Accuracy.
Evaluates model performance on the validation set.

Translation testing:
Allows input of custom sentences for translation.
Sample Outputs
After running the notebook:

The best fine-tuned model is saved in the ./best_fine_tuned_bart/ directory. 
BLEU and other metrics will be displayed for validation performance.

Extra notes:
The /best_fine_tuned_bart/ directory is not included here, but all of the contents are uploaded aside from Models.safetensors because of the file size. 
The adjustable parameters include the following:
Batch size- modify in train_loader (default: 8).
Epochs -change epochs = 3 to desired number.
Learning rate- adjust lr=5e-5 in the AdamW optimizer.
Early stopping patience- change patience = 3.
