import pydantic
from pydantic_docx import Docx_Paragraph_and_Runs, read_docx #type:ignore
import re
import logging
from itertools import chain, compress
from typing import Optional, Dict, List, Any, Union, Tuple
import string
import pickle



logger = logging.getLogger(__name__)

def monolith_root_and_lemma_processor(parsed_object_list, char_counts,verbose = False) -> Dict[str,Any]:
   para_text_lookup = {}
   root_ind_list = []
   subroot_ind_list = []
   lemma_ind_list = []
   some_error_ind_list = []
   reject_ind_list = []
   root_and_lemma_one_line = []
   root_lookup = {}
   lemma_lookup = {}
   # pos_lookup = {}
   # pos_ind_list = []

   up_alph_chars = [x.upper() for x in char_counts.keys() if x.upper() != x.lower()] #only uppercase alphabetical chars

   for i, entryObj in parsed_object_list:

      try:
         para_text_lookup[i] = entryObj.para_text
         successful_cleaner_output:bool = entryObj.cleaner() #by default cleaner removes leading whitespace and merges adjacent runs with identical format features
         # print('sucessful cleaner')
         if not successful_cleaner_output:
            reject_ind_list.append(i)
            print(f'para# {i} IS ONLY whitespace. Need to drop it. #TODO')
      except:
         some_error_ind_list.append(f"cleaner error on p: {i}. Text is {entryObj.interogate__para_text()}")
         print('error on cleaner')
         raise
      try:
         root_note_chars = '-+()? ' #characters that encode the author's notes
         sub_root_beginnings = '-+('
         permissive_root_contents = ''.join(list(chain(up_alph_chars,root_note_chars,string.digits)))

         featureConfig = {
         'root': {'docxFeature': 'run_font_size_pt',
                  'strSummary':'fontSize_12.0', 
                  'value':12.0,
                  # 'text_regex_at_feature': root_expression.compile()
                  },
         'subroot': {'docxFeature': 'run_font_size_pt',
                  'strSummary':'fontSize_12.0', 
                  'value':12.0,
                  # 'text_regex_at_feature': subroot_expression.compile()
                  },
         'lemma': {'docxFeature': 'run_bold',
                  'strSummary':'fontBold', 
                  'value':True},
         'lemmaPOS': {'docxFeature': 'run_italic',
                  'strSummary':'fontItalic', 
                  'value':True},
         }
         is_subroot = False
         # return True, (value_mask, run_text), (regex_mask, regex_matches)
         is_root, (root_mask,run_text), _ = entryObj.single_run_feature_identify(featureConfig['root'])
         if is_root:
            rtext = ''.join(chain(compress(run_text,root_mask))).strip()
            root_lookup[i] = (rtext,root_mask,run_text)
            #routine to distinguish between main root and subroot
            for j, r in enumerate(compress(run_text,root_mask)): 
               if j==0:
                  # low_alph_chars = ''.join([x.lower() for x in char_counts.keys() if x.upper() != x.lower()]) #only uppercase alphabetical chars
                  # up_alph_chars = ''.join([x.upper() for x in char_counts.keys() if x.upper() != x.lower()]) #only uppercase alphabetical chars #type: ignore
                  root_note_chars = '-+()? ' #characters that encode the author's notes
                  sub_root_beginnings = '-+('
                  permissive_root_contents = ''.join(list(chain(up_alph_chars,root_note_chars,string.digits)))
                  pattern = '^['+re.escape(sub_root_beginnings)+']['+re.escape(permissive_root_contents)+']+'
                  m = re.search(pattern = pattern, string = r)
                  if m is not None:
                     is_subroot = True
            if is_subroot:
               # print('\n\nsubroot at para number: ',i)
               # paraText = entryObj.interogate__para_text()
               # print('\t',paraText)
               subroot_ind_list.append(i)
            else:
               # print('\n\nroot at para number: ',i)
               # paraText = entryObj.interogate__para_text()
               # print('\t',paraText)
               root_ind_list.append(i)
         # return True, (value_mask, run_text), (regex_mask, regex_matches)
         is_lemma, (lemma_mask, run_text), _ = entryObj.single_run_feature_identify(featureConfig['lemma'])
         if is_lemma:
            # print(lemma_mask,run_text)
            ltext = ''.join(chain(compress(run_text,lemma_mask))).strip()
            # print(ltext)
            lemma_lookup[i] = (ltext, lemma_mask,run_text)
            # paraText = entryObj.interogate__para_text()
            # print('\t\tp#',i,'\t\t',paraText)
            lemma_ind_list.append(i)
         # is_pos, (pos_mask, run_text), _ = entryObj.single_run_feature_identify(featureConfig['lemma'])
         # if is_pos:
         #    postext = ''.join(chain(compress(run_text,pos_mask))).strip()
         #    # noun_patterns = [r'n\.',r"(?<=\()[^\)]+",r"\/"]
         #    # nounPatternRegex = re.compile('|'.join([p for p in noun_patterns]))
         #    pos_lookup[i] = (postext,pos_mask)
         #    pos_ind_list.append(i)
         #    if is_pos and not is_lemma:
         #       entryObj.paragraph_logger(level = 10,msg = f'Unexpected structure:: para#{i} has a POS: {postext} but this para does NOT have a lemma. last lemma at para#{lemma_ind_list[-1]} was "{lemma_lookup[lemma_ind_list[-1]]}"',print_bool = True)
         # if is_pos and (is_root or is_subroot):
         if is_lemma and is_root:
            # print(f'this para# {i} has BOTH lemma AND root')
            root_and_lemma_one_line.append(i)

      except BaseException as e:
         
         some_error_ind_list.append(i)
         raise e

   if verbose:
      print('roots: ',len(root_ind_list))
      print('subroots: ',len(subroot_ind_list))
      print('lemmas: ',len(lemma_ind_list))
      print('root_and_lemma_one_line: ',len(root_and_lemma_one_line))
      print('additional cleaner rejects: ',len(reject_ind_list))
      print('additional error rejects: ',len(some_error_ind_list))
      # # Test messages
      logger.debug("logger debug test")
      logger.info("Just an information")
         # logger.warning("Its a Warning")
         # logger.error("Did you try to divide by zero")
         # logger.critical("Internet is down")
      print('num entities: ',len(root_ind_list) + len(lemma_ind_list) + len(subroot_ind_list))
      num_good_paras_of_other_content= len(root_ind_list) + len(lemma_ind_list) - len(root_and_lemma_one_line) + len(subroot_ind_list)\
                                          + len(reject_ind_list) + len(some_error_ind_list)
      print('num_good_paras_of_other_content: ',num_good_paras_of_other_content)

   outcomes_dict = {}
   outcomes_dict['parsed_object_list'] = parsed_object_list
   outcomes_dict['para_text_lookup'] = para_text_lookup
   outcomes_dict['root_ind_list'] = root_ind_list
   outcomes_dict['subroot_ind_list'] = subroot_ind_list
   outcomes_dict['lemma_ind_list'] = lemma_ind_list
   outcomes_dict['root_and_lemma_one_line'] = root_and_lemma_one_line
   outcomes_dict['root_lookup'] = root_lookup
   outcomes_dict['lemma_lookup'] = lemma_lookup
   # outcomes_dict['pos_lookup'] = pos_lookup
   outcomes_dict['char_counts'] = char_counts
   return outcomes_dict

# def create_frame(content: Dict[str,Any], heirarchy:List[str]):


if __name__ == '__main__':
   # docx_filename = "Fula_Dictionary-repaired.docx"
   docx_filename = "pasted_docx page 1.docx"

   parsed_to_dict = read_docx(docx_filename)

   outcomes_dict = monolith_root_and_lemma_processor(parsed_to_dict['parsed_object_list'],parsed_to_dict['char_counts'])
   
   parsed_object_list = outcomes_dict['parsed_object_list'] 
   para_text_lookup = outcomes_dict['para_text_lookup'] 
   root_ind_list = outcomes_dict['root_ind_list'] 
   subroot_ind_list = outcomes_dict['subroot_ind_list'] 
   lemma_ind_list = outcomes_dict['lemma_ind_list'] 
   root_and_lemma_one_line = outcomes_dict['root_and_lemma_one_line'] 
   root_lookup = outcomes_dict['root_lookup'] 
   lemma_lookup = outcomes_dict['lemma_lookup'] 
   char_counts = outcomes_dict['char_counts']


