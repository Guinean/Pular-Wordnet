# altair to visualize the data. read in pickle
# determine df shape for this to work
# import birdseye and snoop
from ast import Return
from gzip import READ
from operator import attrgetter
from typing_extensions import runtime
import snoop
import altair as alt
# from vega_datasets import data
from typing import Optional, Dict, List, Any, Union, Tuple, Set
import pydantic
from pydantic import ValidationError, validator, root_validator, Field, constr, BaseModel
from pydantic_docx import Docx_Paragraph_and_Runs, read_docx, extract_features #type:ignore
from pydantic_docx_processor import create_sized_dataframe, expand_dataframe #type:ignore
import re
import json
from itertools import compress, chain
from datetime import datetime
import pandas as pd
import numpy as np
from functools import partial
import pickle



class complex_masked_strings(pydantic.BaseModel):
   checked : bool
   present : Optional[bool]
   strings : Optional[List[str]]
   masks : Optional[np.ndarray]
   #validate assignment, ensure lengths
   #functions for regex that update submasks
   #functions for regex that only check whole runs

   class Config:
      validate_all = True
      validate_assignment = True
      # smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'

   @validator('masks')
   def validate_strings_masks(cls,v,values):
      if 'strings' in values and v != values['strings']:
         raise ValueError('strings and mask do not match')
      current_state = values.get('current_state',False)
      if current_state:
         if 'strings' in current_state and v != current_state['strings']:
            raise ValueError('strings and mask do not match')
      return v

class masked_chars(pydantic.BaseModel):
   chars: np.ndarray
   bools : np.ndarray

   class Config:
      validate_all = True
      validate_assignment = True
      # smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'

   @validator('chars')
   def validate_chars(cls,v,values):
      assert isinstance(v,np.ndarray)
      assert v.dtype == 'str'
      return v

   @validator('bools')
   def validate_bools(cls,v,values):
      assert isinstance(v,np.ndarray)
      assert v.dtype == 'bool_'
      return v


# class dict_glosses(pydantic.BaseModel):

class pydantic_lemma(pydantic.BaseModel):
   index: int 
   paragraph: Docx_Paragraph_and_Runs
   text: str = Field(...,min_length = 1) #article
   line_runs: List[str]
   irregularities : Optional[Dict[str,str]]

      #lemma derivative
   unusedContent: Optional[List[Tuple[np.ndarray,np.ndarray]]] #= Field(...) #int is index for run to allow tracing as entries are removed
   
   FulaDialects: Optional[Set[str]]  #article
   FulaPOSTags: Optional[List[str]]  #article
   FulaPOSClass: Optional[List[str]]  #Noun, Adj, Verb, Adv, Prn, ..., Complicated, Indeterminate
   FulaNoun_Classes : Optional[List[str]]
   FulaNoun_Plurals : Optional[List[str]]
   # FulaNoun_NounsAndClass: Optional[List[Tuple[str,str]]]  #Pular has unique noun classes. Lemma will be copied here beside its class ("noun", "nounclass"), and any additional singular forms will be in additional tuples in this same list
   # FulaNoun_PluralsAndClass: Optional[List[Tuple[str,str]]]  #temporary measure #TODO. include list of wordclasses find all and not '/' findall #Optional[List[Tuple[str,str]]  #Dict provides plurals for nouns. Tuples will have ("noun", "nounclass")
   FulaSynonyms: Optional[List[str]]  #article
   FulaCrossRef: Optional[List[str]] 
         # FulaVerbClass:
   class Config:
      validate_all = True
      validate_assignment = True
      smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'

   def parse_lemmaLine(self):
      
      pos_config = {'docxFeature': 'run_italic',
               'strSummary':'fontItalic', 
               'value':True}
      (is_target, text, target_mask, run_text) = extract_features(self.paragraph,pos_config)
      #POS
      if is_target:
         assert not isinstance(run_text,bool)
         assert not isinstance(target_mask,bool)
         matches = list(compress(run_text,target_mask))
         try:
            text = ';'.join(matches) #TODO this only pulls POS info from the first contiguous italics run
            if len(matches)>1:
               print(self.paragraph.run_text)
         except:
            raise
         #noun
            #regex () to remove FulaWordClass
            #regex / to get plural
            # regex n.
         self.FulaPOSTags = text.replace('+',',').replace(';',',').split(',')
         word_class = re.findall(r"\(([^\)]+)\)",text) #anything between parenthesis
         is_verb = bool(re.findall(r"v.",text.lower()))
         noun_flag = re.findall(r"(n\.)",text) #find n.        
         plurals = re.findall(r"([^\/]+)",text) #anything not a /
         #TODO plurals doesnt work with leading /, and will miss any data from cleaning issues (not italic slash)
         if any([len(word_class)>0, noun_flag, len(plurals)>1]):
            self.FulaNoun_Classes = word_class
            self.FulaNoun_Plurals = plurals
            # # print('here')
            # is_noun = True
            self.FulaPOSClass = ['Noun']
            # if word_class:
            #    self.FulaNoun_NounsAndClass = [(self.text,word_class[0])]
            # if len(plurals)>1:
            #    # self.FulaNoun_NounsAndClass
            #    if len(plurals)>2:
            #       print(self.index,' - index lemma has multiple noun slashes "/"')
            #    # plurals_word_class = re.findall("\(([^\)]+)\)",plurals[1])  #anything between parenthesis
            #    # plurals_text = re.findall("(?<![^\)])[^\(]+(?![^\(]*\))",plurals[1]) #anything NOT between ()
            #    # self.FulaNoun_PluralsAndClass = plurals #TODO
            # else:
            #    pass
         elif is_verb:
            # self.FulaNoun_NounsAndClass = False
            self.FulaPOSClass = ['Verb']
         else:
            self.FulaPOSClass = ['Other_']
      else:
         self.FulaPOSClass = ['None']
            

      #dialect
      dialects = []
      for run in self.line_runs:
         dialects.extend(re.findall(r"\<([^\>]+)\>",run)) #anything between parenthesis
      if len(dialects)>0:
         self.FulaDialects = set(dialects)
      else:
         self.FulaDialects = set(['Mali'])
      #cross ref
         # cross_reference = r"cf\.\:(.+)|cf\:(.+)" #TODO
      #syn
         #TODO

      
      return

   def parse_noun_class(self):
      assert self.FulaPOSClass == 'Noun'
      para = self.paragraph
      before_helv = ''.join(chain(para.run_text[:para.run_font_name.index('Helv 8pt')]))
      try:
         singles,plurals = before_helv.split('/')
      except ValueError:
         singles = before_helv.split('/')[0]

      def noun_classes(string):
         classes = re.findall(r"\(([^0-9\)]+)\)",string)
         nclass_groups = re.split(r"\(([^0-9\)]+)\)",string)
         nclass_groups = [i for i in nclass_groups if len(i.strip())>0 and i not in classes]
         nclass_groups = [[ii.strip() for ii in i.split(',')] for i in nclass_groups]
         # nclass_groups = [i.split(',') for i in nclass_groups]
         print(nclass_groups)
         assert len(nclass_groups) == len(classes), f"{nclass_groups},\t{classes}"
         return [{'noun_class':c,'word_group':g} for c,g in zip(classes,nclass_groups)]
      try:
         return noun_classes(singles) 
      except:
         return [{'noun_class': np.nan, 'word_group': np.nan}]
      #example output#[{'noun_class': 'o', 'word_group': ['alaada', 'test']}]

   def parse_lemmaLineRemainder(self):
      #TODO
      return


class pydantic_root(pydantic.BaseModel):
   index: int 
   paragraph: Docx_Paragraph_and_Runs
   text: str = Field(...,min_length = 1) #article
   line_runs: List[str]
   origin: Optional[str]
   notes: Optional[str]
   class Config:
      validate_all = True
      validate_assignment = True
      smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'

   def get_rootOrigin(self):
      #TODO
      return


class pydantic_root_subpiece(pydantic.BaseModel):
   index: int 
   paragraph: Docx_Paragraph_and_Runs
   text: str = Field(...,min_length = 1) #article
   line_runs: List[str]
   notes: Optional[str]
   class Config:
      validate_all = True
      validate_assignment = True
      smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'


class dict_entry(pydantic.BaseModel): 
   '''
   #True is used as default value, and must be changed to False if the feature is not found. This will catch not implemented, and incomplete updates
   '''
   #paras
   subsidiary_paragraphs: Optional[List[Docx_Paragraph_and_Runs]] #= Field(min_items=1)
   # unused_content : List[np.ndarray]
   irregularities: List[str] = []
   root : pydantic_root
   root_subpiece : Optional[pydantic_root_subpiece]
   lemma: pydantic_lemma

   class Config:
      validate_all = True
      validate_assignment = True
      smart_union = True 
      arbitrary_types_allowed = True 
      extra = 'forbid'

   def parse_senses(self):
      #TODO
      return

   def parse_glossRemainder(self):
      #TODO
      return

   def separate_codeclared_roots_and_lemmas(self):
      #TODO
      return


def parse_dataframe_groups(workinglemmas, parsed_object_lookup, eDP:pd.DataFrame):
   #entityDeclarationParas
   all_entries = []
   for indexes,para_inds in workinglemmas.items():
      all_values = {}
      all_values['subsidiary_paragraphs'] = [parsed_object_lookup[i] for i in para_inds]
      root_index,root_subpiece_index,lemma_index = indexes

      for key,ind in {'lemma':lemma_index,'root':root_index,'root_subpiece':root_subpiece_index}.items():
         values = {}
         if not np.isnan(ind):
            # print(k,v)
            values['index'] = ind
            values['paragraph'] = parsed_object_lookup[ind]
            values['text'] = eDP.at[int(ind),'text_any_entity']
            values['line_runs'] = eDP.at[int(ind),'run_text_list_any_entity']
            all_values[key]=values
         else:
            pass
      this_entry = dict_entry(**all_values)
      this_entry.lemma.parse_lemmaLine()

      all_entries.append(this_entry)

   return all_entries



if __name__ == '__main__':
   import logging
   import os
   script_dir = os.path.dirname(__file__)

   now = datetime.now()
   current_time = now.strftime("%Y-%m-%d_-_%H-%M-%S")
   logger_filename = f"logs_and_outputs/{current_time}docxFileParse.log"

   handler = logging.FileHandler(logger_filename, 'w', 'utf-8') 
   handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

   # logging.setLogRecordFactory(factory)
   logging.basicConfig(handlers=[handler], level=logging.DEBUG)
   logger = logging.getLogger()


   with open('pickled_results/feature_Frames_and_Indexes.pkl', 'rb') as file:
      # Call load method to deserialze
      output = pickle.load(file, encoding='utf-8')
   (
      parsed_object_list, #:List[Tuple[int,Docx_Paragraph_and_Runs]]
      parsed_object_lookup, #:Dict[int,Docx_Paragraph_and_Runs] = dict(parsed_object_list)
      doc_para_count, #: int = int(parsed_to_dict['total_encountered_paragraphs']) #type: ignore
      char_counts, #: Counter = parsed_to_dict['char_counts'] #type: ignore
      rootFrame,#:pd.DataFrame
      rootsubpieceFrame, #:pd.DataFrame
      lemmaFrame, #:pd.DataFrame
      nonentityParaFrame, #pd.DataFrame
      cleanerOutcomesDf #pd.DataFrame
   ) = output

   paratext_lookup = {k:v.interogate__para_text() for k,v in parsed_object_lookup.items()}

   with open('pickled_results/all_inheritance_frames_tup.pkl', 'rb') as file:
      # Call load method to deserialze
      output = pickle.load(file, encoding='utf-8')
   # print(output)
   (
      allParagraphsInheritanceFrame,
      entityDeclarationParas,
      paragraphRecordsFrame
   ) = output
   # print(type(allParagraphsInheritanceFrame))
   all_inheritance_frames_dict = {
      'allParagraphsInheritanceFrame':allParagraphsInheritanceFrame,
      'entityDeclarationParas':entityDeclarationParas,
      'paragraphRecordsFrame':paragraphRecordsFrame
   }

   workinglemmas = paragraphRecordsFrame.groupby(['paragraphIndex_root','paragraphIndex_root_subpiece','paragraphIndex_lemma'])
   ct_all_lemmas_with_dependent_paragraphs = len(workinglemmas)
   print(ct_all_lemmas_with_dependent_paragraphs)
   workinglemmas = {k:list(v) for k,v in workinglemmas.groups.items() if len(v) >=2}
   ct_all_lemmas_with_2plus_dependnet_paragraphs = len(workinglemmas)
   print(ct_all_lemmas_with_2plus_dependnet_paragraphs)
   # sorted(list(workinglemmas.items()))[:5]

   all_entries = parse_dataframe_groups(workinglemmas, parsed_object_lookup, entityDeclarationParas)
   print(len(all_entries))
   assert all_entries is not None

   from collections import Counter
   # pos_counter = Counter()
   pos_list = []
   noun_entries = []
   verb_entries = []
   for e in all_entries:
      pos_list.extend(list(e.lemma.FulaPOSClass))

   print(Counter(pos_list))

   print(all_entries[0])
   
   # print("number of lemma-identified paragraph clusters associated with 2 or more dependent content paragraphs: ",ct_all_lemmas_with_2plus_dependnet_paragraphs)
   # _ = [print('\t',k,v) for k,v in dictionary_numbers_dict.items()]


# codeclared lemmas and roots                1438 #<<<
# instances of non-lemma bold lines          40   #<<<

# true lemmas: 9610 - 40 =                   9570 #<<<
# true lemmas not co declared 9570 - 1438              
# lemmas with any paragraphs following       7994

# # import birdseye as eye
# %load_ext birdseye
# https://birdseye.readthedocs.io/en/latest/integrations.html

# ...Preparing Pickle...
# root_subpiece                              713
# root                                       6381
# lemmas                                     9610
# num entities in the docx:                  16704
# num of overlapping entities in the docx:   1438
# paragraphs in the docx:                    32507
# non-entity paragraphs in the docx:         17241

# co_declared_lemmas = 1438
# bold_lines = 9610
# non_leading_bold_lines = 40

# all_lemmas = bold_lines - non_leading_bold_lines
# print(all_lemmas)
# non_codeclared_lemmas = all_lemmas - co_declared_lemmas
# print(non_codeclared_lemmas)
# lemmas_with_paragraph_content = 7994
# lemmas_with_NO_paragraph_content = non_codeclared_lemmas - lemmas_with_paragraph_content
# print(lemmas_with_NO_paragraph_content)
# lemmas_with_TWOPLUS_paragraph_of_content = 7985
# lemmas_with_ONE_paragraph_of_content = 9

# final_usable_lemma_count = lemmas_with_TWOPLUS_paragraph_of_content #7985


# From the Kamusi Article
# >When parsing was completed, the source dictionary resolved to 7918 Fula entries and 10970 Fula senses.

# Current output of my parser pipeline?
# > my parser was able to find **67 more** entries than their parsing routine.
# > 
# > Roots word pieces:    6381
# >
# > Root subword pieces: 713
# > 
# > Lemmas:   7985 usable for two-way Wordnet Matching described in the article (9570 total)

   import os
   pickle_filename = os.path.join(script_dir, "pickled_results/all_dictionary_entries_list_test.pkl")
   with open(pickle_filename, 'wb') as file:
         pickle.dump(all_entries, file)
# with open('pickled_results/all_dictionary_entries_list.pkl', 'wb') as file:
#    pickle.dump(all_entries, file)
