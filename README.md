# Pular-Wordnet

Capstone Project for UMICH Data Science Masters Degree

## What's going on here?

This is the repository for my Capstone project. 
The goal of this repository is to create NLP tools to assist in collecting / labeling data for under-resourced languages, specifically in West Africa.

## What might this become?
Eventually I hope to expand this to an MS Word extensions (and LibreOffice) to provide tools I take for granted, dictionary, grammer check, thesaurus, etc. This will also allow greater access to user feedback, offline, private, and at the owners discretion to package and share.

## But what is it NOW?
Currently, this repo provides type-validated wrapping classes over python-docx and docx data objects. These wrapper classes provide loggable methods for processing and cleaning the contained XML / python-docx data. This script may be found in **pydantic_docx.py**

the **pydantic_docx_processor.py** scripts provides utility functions to invoke the pydantic_docx script on a target document, and return the resulting data as a indexed list with multiple pandas dataframes containing pointers the class objects in question, as well as some preliminary metadata extracted.

And example pipeline that leverages this may be found in the jupyter notebooks 01 and 02 in the main directory.

Example data is provided in the **test_data** folder.

The **pickled_results** folder is used to save intermediate representation created in this process. If used, please open with caution and at you own risk, recognizing the normal risks with opening any pickled object. I trust these, but make your own judgements.
Eventually I will need to find other less questionable ways to save custom classes like these

The **logs and outputs** folder will recieve logs created during running. Not the gitignore settings regarding logs

The **demonstrations** folder contains saved version of note, as well as images and other files used in creating the Article found in **Writings**

the **data_handling** folder will be the beginnings of a new NetworkX API (instead of pandas) for the package, but currently the folder just uses NetworkX to visualize the python-docx data objects.


