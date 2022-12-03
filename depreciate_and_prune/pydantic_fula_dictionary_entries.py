# altair to visualize the data. read in pickle
# determine df shape for this to work
# import birdseye and snoop
from ast import Return
from gzip import READ
from typing_extensions import runtime
import snoop
import altair as alt
# from vega_datasets import data
from typing import Optional, Dict, List, Any, Union, Tuple
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

def getAssert(dct,val,typ:type, strict = True) -> Any:
   out = dct.get(val,None)
   if strict and typ is type(None):
      raise NotImplementedError(f'this get/assert function cannot "get" None values safely, and a {val} type was passed for dict {dict}')
   if strict and out is None:
      raise KeyError(f"the provided dict {dct} did not have the key: {val}")
   assert isinstance(out,typ), f"the provided dict {dct} did not give {val} with the expected type: {typ}, but instead {out}"
   return out
   
def trustyGet(obj:Docx_Paragraph_and_Runs, feat: str, silent_return = True) -> str:
      if not silent_return:
         output : str = getattr(obj,feat,'')
         if len(output) > 0:
            return output
         else:
            raise AttributeError(f'trusty getter could not find: {feat}')
      else:
         output : str = getattr(obj,feat,'')
         return output
class dict_entry(pydantic.BaseModel): 
   '''
   #True is used as default value, and must be changed to False if the feature is not found. This will catch not implemented, and incomplete updates
   '''
   irregularities: List[str] = []
   lemma_index: int 
   
   lemma: str = Field(...,min_length = 1) #article
   root: str
   root_subpiece: Union[str,bool] = True #these will not be updated after initialization
   root_metadata: Dict[str,Any]  #index,root_text, root_meta_data #article 
   root_origin: Union[str,bool] = True #these will not be updated after initialization
   #Paragraph Derivatives: Glosses and Annotations
   lemmaLine_runs: List[str] #these will not be updated after initialization
   englishGlossLine_runs: Union[List[str],bool] = True #these will not be updated after initialization
   frenchGlossLine_runs: Union[List[str],bool] = True #these will not be updated after initialization
   FulaAnnotations_runs: Union[List[List[str]],bool]= True #these will not be updated after initialization

   #Gloss Derivatives: English Senses
   FulaSenseEnglish: Union[List[str],bool] = True #list of "senses" split by semicolons #article
   FulaSenseEnglish_Count: Union[int,bool] = True #number of senses in english #aka FulaSenseClassifications #article
   FulaSenseEnglish_Annotations: Union[List[str],bool] = True #contains the annotations (in parenthesis) for a given sense, if any #article 
   FulaSenseEnglish_Synonyms: Union[List[str],bool] = True #holds bracket text for a sense, suspected all synonyms. These may all occur at the end, and may be redundant with the synonyms provided at the head of the entry
   FulaSenseEnglish_unusedContent: Union[List[str],bool] = True #holds any run text that is not contained in the above features
   
   #Gloss Derivatives: French Senses
   FulaSenseFrench: Union[List[str],bool] = True #article
   FulaSenseFrench_Count: Union[int,bool] = True #aka FulaSenseClassifications
   FulaSenseFrench_Annotations: Union[List[str],bool] = True
   FulaSenseFrench_Synonyms: Union[List[str],bool] = True
   FulaSenseFrench_unusedContent: Union[List[str],bool] = True
   
   #lemma derivative
   lemmaLine_unusedContent: List[Tuple[int,str]] #= Field(...) #int is index for run to allow tracing as entries are removed
   FulaDialects: Union[List[str],bool] = True #article
   FulaPOSTags: Union[List[str],bool] = True #article
   FulaPOSClass: Union[List[str],bool] = True #Noun, Adj, Verb, Adv, Prn, ..., Complicated, Indeterminate
   FulaNoun_NounsAndClass: Union[List[Tuple[str,str]],bool] = True #Pular has unique noun classes. Lemma will be copied here beside its class ("noun", "nounclass"), and any additional singular forms will be in additional tuples in this same list
   FulaNoun_PluralsAndClass: Union[List[str],bool] = True #temporary measure #TODO. include list of wordclasses find all and not '/' findall #Union[List[Tuple[str,str]],bool] = True #Dict provides plurals for nouns. Tuples will have ("noun", "nounclass")
   FulaSynonyms: Union[List[str],bool] = True #article
   FulaCrossRef: Union[List[str],bool] = True
         # FulaVerbClass:
   #paras
   paragraphs: List[Docx_Paragraph_and_Runs] = Field(...,min_items = 1)
   
   class Config:
      validate_all = True
      # validate_assignment = True
      smart_union = True  
      extra = 'forbid'

   @root_validator(pre=True)
   def _validate_and_build(cls, values: Dict[str,Any]) -> Dict[str, Any]:
      # print(values)
      newValues: Dict[str,Any] = {}
      newValues['irregularities'] = []

      ###PARAGRAPHS CHECK###
      # print(values)
      newValues['paragraphs'] = getAssert(values,'paragraphs',list) #these come in as tuples(ind,para)
      #TODO have more rigorous checks on para since so much depends on that even in this validation
      assert isinstance(newValues['paragraphs'][0],tuple), 'currently paras must be packaged in a tuple with their original doc index'
      newValues['paragraphs'] = [p for i,p in newValues['paragraphs']]
      if len(newValues['paragraphs']) == 0:
         raise ValueError('given invalid paragraphs of zero length')
      else: #logic for subsidary paragraphs
         lemma_line = trustyGet(newValues['paragraphs'][0],feat = 'run_text', silent_return=False)
         newValues['lemmaLine_runs'] = lemma_line
         if len(newValues['paragraphs']) <3:
            raise ValueError('incorrect number of paragraphs provided')
         elif len(newValues['paragraphs']) == 3:
            newValues['englishGlossLine_runs'] = newValues['paragraphs'][1].get_run_text()
            newValues['frenchGlossLine_runs'] = newValues['paragraphs'][2].get_run_text()
         else: 
            newValues['englishGlossLine_runs'] = newValues['paragraphs'][1].get_run_text()
            newValues['frenchGlossLine_runs'] = newValues['paragraphs'][2].get_run_text()
            newValues['FulaAnnotations_runs'] = []
            for p in newValues['paragraphs'][3:]:
               # print(p.get_run_text())
               newValues['FulaAnnotations_runs'].append(p.get_run_text())
          
      try: ###LEMMA CHECK###
         #reading in lemma results from first pass of pydantic parser
         lemma_index, lemma_mask, lemmaLine_runs = getAssert(values,'lemma',tuple)
         #checking for correct structure
         lemma_matched_runs = list(compress(lemmaLine_runs,lemma_mask))
         if len(lemma_matched_runs) > 1:
            newValues['irregularities'].append('What:Unexpected, Where: Lemmas, Why: the merge routines should aggregate adjacent runs with same features. Multiple Lemma runs should not be possible if Bold is contiguous')
         lemma_text = ''.join(chain(lemma_matched_runs)).strip()
            #TODO have better control for expected structure that these runs should be adjacent (and only should be one)
         #saving values. These will have to pass type check of declared attribute types for validation to succeed
         newValues['lemma'] = lemma_text
         newValues['lemma_index'] = lemma_index
         newValues['lemmaLine_runs'] = lemmaLine_runs
         # first_lemma = False
         used_run_mask = [False]*len(lemma_mask)
         for i,b in enumerate(lemma_mask):
            if b:
               used_run_mask[i]=True

      except Exception as e:
         #exception lemma check
         raise RuntimeError('failed lemma check') from e

      try: ###ROOT CHECK###
         #reading in root results from first pass of pydantic parser
         #checking for correct structure
         root_index, root_mask, rootLine_runs = getAssert(values,'root',tuple)
         root_matched_runs = list(compress(rootLine_runs,root_mask))
         if len(root_matched_runs) > 1:
            newValues['irregularities'].append('What:Unexpected, Where: Root, Why: the merge routines should aggregate adjacent runs with same features. Multiple Root runs should not be possible if Fontsize is contiguous and unique')
         root_text = ''.join(chain(root_matched_runs)).strip()
         if root_index == newValues['lemma_index']:
            newValues['irregularities'].append('What:inconsistent, Where: Lemmas and Root, Why:normally root and lemma are on different lines. This has them sharing, which may indicate a lack of other content')
            used_run_mask = [any(run) for run in list(zip(used_run_mask,root_mask))]
            rootLine_runs = False
         newValues['root'] = root_text
         newValues['root_metadata'] = {'root_index': root_index, 'root_runs':rootLine_runs}
      except Exception as e:
         #exception ROOT check
         raise RuntimeError('failed ROOT check') from e

      try: ###ROOT-Subpiece CHECK###
         #iterating in case a subroot is present
         root_index, root_mask, rootLine_runs = getAssert(values,'root_subpiece',tuple)
         root_matched_runs = list(compress(rootLine_runs,root_mask))
         if len(root_matched_runs) > 1:
            newValues['irregularities'].append('What:Unexpected, Where: Root-Subpiece, Why: the merge routines should aggregate adjacent runs with same features. Multiple Root runs should not be possible if Fontsize is contiguous and unique')
         root_subpiece_text = ''.join(chain(root_matched_runs)).strip()
         if root_index == newValues['lemma_index']:
            newValues['irregularities'].append('What:inconsistent, Where: Lemmas and Root-Subpiece, Why:normally root and lemma are on different lines. This has them sharing, which may indicate a lack of other content')
            used_run_mask = [any(run) for run in list(zip(used_run_mask,root_mask))]
            rootLine_runs = False
         newValues['root_subpiece'] = root_subpiece_text
         for k,v in {'root_subpiece_index': root_index, 'root_subpiece_runs':rootLine_runs}.items():
            newValues['root_metadata'][k] = v
      except AssertionError as e:
         newValues['root_subpiece'] = False
      except Exception as e:
         #exception ROOT-Subpiece check
         raise RuntimeError('failed ROOT-Subpiece check') from e

      #determining lemmaLine_unusedContent
      unused_run_mask = [not r for r in used_run_mask]
      newValues['lemmaLine_unusedContent'] = list(compress(enumerate(newValues['lemmaLine_runs']),unused_run_mask))
      return newValues

   def parse_senses(self):
      #TODO
      return

   def parse_glossRemainder(self):
      #TODO
      return

   def parse_lemmaLine(self):
      
      pos_config = {'docxFeature': 'run_italic',
               'strSummary':'fontItalic', 
               'value':True}
      (is_target, text, target_mask, run_text) = extract_features(self.paragraphs[0],pos_config)
      #POS
      if is_target:
         assert not isinstance(run_text,bool)
         assert not isinstance(target_mask,bool)
         matches = list(compress(run_text,target_mask))
         try:
            text = matches[0] #TODO this only pulls POS info from the first contiguous italics run
         except:
            print(text)
            print(run_text)
            print(matches)
            raise RuntimeError()
         #noun
            #regex () to remove FulaWordClass
            #regex / to get plural
            # regex n.
         word_class = re.findall(r"\(([^\)]+)\)",text) #anything between parenthesis
         # print(word_class)
         noun_flag = re.findall(r"(n\.)",text) #find n.
         plurals = re.findall(r"([^\/]+)",text) #anything not a /
         if any([len(word_class)>0, noun_flag, len(plurals)>1]):
            # print('here')
            is_noun = True
            self.FulaPOSClass = ['Noun']
            if word_class:
               self.FulaNoun_NounsAndClass = [(self.lemma,word_class[0])]
            if len(plurals)>1:
               # self.FulaNoun_NounsAndClass
               if len(plurals)>2:
                  print(self.lemma_index,' - index lemma has multiple noun slashes "/"')
               # plurals_word_class = re.findall("\(([^\)]+)\)",plurals[1])  #anything between parenthesis
               # plurals_text = re.findall("(?<![^\)])[^\(]+(?![^\(]*\))",plurals[1]) #anything NOT between ()
               self.FulaNoun_PluralsAndClass = plurals #TODO
            else:
               pass
         else:
            self.FulaNoun_NounsAndClass = False
            is_verb = bool(re.findall(r"v.",text.lower()))
            if is_verb:
               self.FulaPOSClass = ['Verb']
            self.FulaPOSTags = text.replace('+',',').split(',')

      #dialect
      dialects = []
      for run in self.lemmaLine_runs:
         dialects.extend(re.findall(r"\<([^\>]+)\>",run)) #anything between parenthesis
      if len(dialects)>0:
         self.FulaDialects = list(set(dialects))
      else:
         self.FulaDialects = ['Mali']
      #cross ref
         # cross_reference = r"cf\.\:(.+)|cf\:(.+)" #TODO
      #syn
         #TODO

      
      return

   def parse_lemmaLineRemainder(self):
      #TODO
      return

   def get_rootOrigin(self):
      #TODO
      return
 
   # def give_entryText(self, joiner = '\t') -> str: #article
      # return joiner.join([para.trustyGet('para_text') for para in self.paragraphs])

   
      
def parse_dataframe_groups(workinglemmas, parsed_object_lookup, entityDeclarationParas):

   all_entries = []
   # print(len(workinglemmas.keys()))
   for e in workinglemmas.items():
      values = {}
      root,root_subpiece,lemma = e[0]
      paras = e[1]
      # parsed_object_lookup[root]
      values['paragraphs']= [int(lemma)]+paras
      values['paragraphs'] = [(i,parsed_object_lookup[i]) for i in values['paragraphs']]
      # print(len(values['paragraphs']))
      lemma_mask = entityDeclarationParas.loc[int(lemma),:]['mask_any_entity']
      lemmaLine_runs = entityDeclarationParas.loc[int(lemma),:]['run_text_list_any_entity']
      values['lemma'] = (int(lemma),lemma_mask,lemmaLine_runs)
      root_mask = entityDeclarationParas.loc[int(root),:]['mask_any_entity']
      rootLine_runs = entityDeclarationParas.loc[int(root),:]['run_text_list_any_entity']
      values['root'] = (int(root),root_mask,rootLine_runs)
      if not np.isnan(root_subpiece):
         root_subpiece_mask = entityDeclarationParas.loc[int(root_subpiece),:]['mask_any_entity']
         root_subpieceLine_runs = entityDeclarationParas.loc[int(root_subpiece),:]['run_text_list_any_entity']
         values['root_subpiece'] = (int(root_subpiece),root_subpiece_mask,root_subpieceLine_runs)
      else:
         values['root_subpiece'] = False
      # print(values.get('paragraphs'))
      this_entry = dict_entry(**values)
      # print('here')
      # print(this_entry.json(indent=3))
      this_entry.parse_lemmaLine()
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
   # print(len(all_entries))
   assert all_entries is not None

   noun_entries = []
   for e in all_entries:
      if e.FulaPOSClass == ['Noun']:
         # print(e)
         noun_entries.append(e)

   verb_entries = []
   for e in all_entries:
      if e.FulaPOSClass == ['Verb']:
         # print(e)
         verb_entries.append(e)
   number_of_noun_entries = len(noun_entries)
   number_of_verb_entries = len(verb_entries)
   number_of_other_entries = len(all_entries)-len(verb_entries)-len(noun_entries)
   number_of_total_entries = len(all_entries)

   dictionary_numbers_dict = {"number_of_total_entries":number_of_total_entries,
   "number_of_noun_entries":number_of_noun_entries,
   "number_of_verb_entries":number_of_verb_entries,
   "number_of_other_entries":number_of_other_entries,
   }

   print("number of lemma-identified paragraph clusters associated with 2 or more dependent content paragraphs: ",ct_all_lemmas_with_2plus_dependnet_paragraphs)
   _ = [print('\t',k,v) for k,v in dictionary_numbers_dict.items()]


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
   pickle_filename = os.path.join(script_dir, "pickled_results/all_dictionary_entries_list.pkl")
   with open(pickle_filename, 'wb') as file:
         pickle.dump(all_entries, file)
# with open('pickled_results/all_dictionary_entries_list.pkl', 'wb') as file:
#    pickle.dump(all_entries, file)
