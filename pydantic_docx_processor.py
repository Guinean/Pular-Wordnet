import pydantic
from pydantic_docx import Docx_Paragraph_and_Runs, read_docx, closest, pairwise, extract_features #type:ignore
import re
import logging
from itertools import chain, compress
from typing import Optional, Dict, List, Any, Union, Tuple
import string
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from collections import Counter


# logger = logging.getLogger(__name__)

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
   normal_para_ind_list = []
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
         if not any([is_lemma, is_root]):
            normal_para_ind_list.append(i)
         

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
   outcomes_dict['normal_para_ind_list'] = normal_para_ind_list

   return outcomes_dict

def create_sized_dataframe(parsed_object_list: List[Tuple[int,Docx_Paragraph_and_Runs]],size: int) -> pd.DataFrame:
   doc_para_index = list(range(size))
   para_indexes = [i for i,p in parsed_object_list]
   df_doc =  pd.DataFrame(data = doc_para_index, index = doc_para_index, columns = ['index'])
   df_parsed = pd.DataFrame(data = para_indexes, index = para_indexes, columns = ['paragraphIndex'])
   
   df_parsed = df_parsed.join(df_doc, how = 'outer', sort=True)
   df_parsed = df_parsed[['paragraphIndex']]
   df_parsed.index.name = 'index'
   # df_parsed
   return df_parsed

def expand_dataframe(df:pd.DataFrame, target:Union[pd.Series,int]) -> pd.DataFrame:
   if isinstance(target,pd.Series):
      target.rename('_target')
      df = df.join(target, how = 'outer', sort=True)
      df.drop('_target',inplace=True)
      return df
   elif isinstance(target,int):
      new_index = list(range(target))
      fillseries =  pd.Series(data = new_index, index = new_index, name = 'index')
      df = df.join(fillseries, how = 'outer', sort=True)
      return df
   else: raise NotImplementedError('no target df sizing condition was provided. Provide a target series or target integer to fill to.')

if __name__ == '__main__':
   import logging
   now = datetime.now()
   current_time = now.strftime("%Y-%m-%d_-_%H-%M-%S")
   logger_filename = f"logs_and_outputs/{current_time}docxFileParse.log"

   handler = logging.FileHandler(logger_filename, 'w', 'utf-8') 
   handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

   # logging.setLogRecordFactory(factory)
   logging.basicConfig(handlers=[handler], level=logging.DEBUG)
   logger = logging.getLogger()



   docx_filename = "Fula_Dictionary-repaired.docx"
   # docx_filename = "pasted_docx page 1.docx"

   parsed_to_dict = read_docx(docx_filename)
   parsed_object_list = parsed_to_dict['parsed_object_list']
   doc_para_count: int = parsed_to_dict['total_encountered_paragraphs'][0] #type: ignore
   char_counts: Counter = parsed_to_dict['char_counts'] #type: ignore
   parsed_object_lookup = dict(parsed_object_list) 
      # outcomes_dict['handled_errors'] = parsed_to_dict['handled_errors']
      # outcomes_dict['failed_paras_ind'] = parsed_to_dict['failed_paras_ind']
   # outcomes_dict = monolith_root_and_lemma_processor(parsed_object_list,parsed_to_dict['char_counts']) #verbose = True
   # with open('parsed_objectClass_outcomes_dict.pkl', 'wb') as file:
   #    pickle.dump(outcomes_dict, file)

   # parsed_object_list = outcomes_dict['parsed_object_list']
   # para_text_lookup = outcomes_dict['para_text_lookup'] 
   # root_ind_list = outcomes_dict['root_ind_list'] 
   # subroot_ind_list = outcomes_dict['subroot_ind_list'] 
   # lemma_ind_list = outcomes_dict['lemma_ind_list'] 
   # # root_and_lemma_one_line = outcomes_dict['root_and_lemma_one_line'] 
   # root_lookup = outcomes_dict['root_lookup'] 
   # lemma_lookup = outcomes_dict['lemma_lookup'] 
   

   cleanerOutcomesDf = create_sized_dataframe(parsed_object_list, doc_para_count) #doc_para_count to have nan where failed paras were
   cleanerOutcomesDf['cleaner_success_outcomes'] = cleanerOutcomesDf['paragraphIndex'] \
      .apply(lambda x: parsed_object_lookup[x].cleaner(execute_defaults=True) if not np.isnan(x) else np.nan)

   up_alph_chars = [x.upper() for x in char_counts.keys() if x.upper() != x.lower()] #only uppercase alphabetical chars
   root_note_chars = '-+()? ' #characters that encode the author's notes
   sub_root_beginnings = '-+('
   permissive_root_contents = ''.join(list(chain(up_alph_chars,root_note_chars,string.digits)))
   root_subpiece_pattern = '^['+re.escape(sub_root_beginnings)+']['+re.escape(permissive_root_contents)+']+'
   featureConfig = {
      'root': {'docxFeature': 'run_font_size_pt',
               'strSummary':'fontSize_12.0', 
               'value':12.0,
               },
      # 'subroot': {'docxFeature': 'run_font_size_pt',
      #          'strSummary':'fontSize_12.0', 
      #          'value':12.0,
      #          # 'root_subpiece_pattern': root_subpiece_pattern, #TODO allow regex within the feature extraction method
      #          # 'permissive_root_contents': permissive_root_contents, #TODO allow regex within the feature extraction method
      #          },
      'lemma': {'docxFeature': 'run_bold',
               'strSummary':'fontBold', 
               'value':True},
      'lemmaPOS': {'docxFeature': 'run_italic',
               'strSummary':'fontItalic', 
               'value':True},
      }


   hierarchy_categories = ['root', 'lemma'] #subroot is an exclusive subclass of root as lexically defined here
   frames_dct: Dict['str',pd.DataFrame] = {}
   for target in hierarchy_categories:
      targetdf = create_sized_dataframe(parsed_object_list,doc_para_count)
      targetdf[[0,1,2,3]] = targetdf['paragraphIndex'].apply( #type: ignore
         lambda x: extract_features(parsed_object_lookup[x],featureConfig[target]) if not np.isnan(x) else np.nan).apply(pd.Series)
      # targetdf.rename(columns = {0:f'is_{target}', 1:f'{target}_text', 2:f'{target}_mask', 3:f'{target}_run_text_list'},inplace=True)
      targetdf.rename(columns = {0:'is', 1:'text', 2:'mask', 3:'run_text_list'},inplace=True)
      #choice point: filtering out untargetted records. Is necessary for current subpiece regex logic below
      # targetdf = targetdf[targetdf[f'is_{target}']==True] #filter for only targetted records
      targetdf = targetdf[targetdf['is']==True] #filter for only targetted records
      targetdf.index.name='index'
      targetdf.name = target
      assert isinstance(targetdf,pd.DataFrame)
      # frames_dct[target] = targetdf.copy()
      if target == 'root':
         root_or_subroot_mask = targetdf[ #type: ignore
               'run_text_list'] \
               .apply(lambda x: x[0]) \
               .apply(lambda x: bool(re.search(root_subpiece_pattern, x)))
         frames_dct['root_subpiece'] = targetdf[root_or_subroot_mask]
         targetdf = targetdf[~root_or_subroot_mask]
         # print('subpiece','c',targetdf.shape)
         frames_dct['root_subpiece'].columns = frames_dct['root_subpiece'].columns.str.replace('is_root', 'is_root_subpiece')
      frames_dct[target] = targetdf.copy()

   
   # print(frames_dct['root_subpiece'].head())
   # dict_df = targetdf.join(
   #          [frames_dct['subroot'],
   #          frames_dct['lemma']],
   #          on = 'index',
   #          how = 'outer'
   #          )
   # print(dict_df.head()) 
   print('...Preparing Pickle...')
   rootFrame = frames_dct['root']
   rootsubpieceFrame = frames_dct['root_subpiece']

   lemmaFrame = frames_dct['lemma']
   for k,v in frames_dct.items():
      print('{:<30s} {}'.format(k, v.shape[0]))
   entityIndexes = rootFrame.index \
      .append(rootsubpieceFrame.index)\
         .append(lemmaFrame.index)
   num_entities = entityIndexes.shape[0]
   print('{:<42} {}'.format('num entities in the docx:  ',num_entities))
   entityIndexes = entityIndexes.drop_duplicates()
   print('{:<42} {}'.format("num of overlapping entities in the docx:", 
                                    num_entities - entityIndexes.shape[0]))

   nonentityParaFrame = create_sized_dataframe(parsed_object_list, doc_para_count)
   print('{:<42s} {}'.format("paragraphs in the docx:", 
                                 nonentityParaFrame.shape[0]))
   nonentityParaFrame = nonentityParaFrame.drop(index = entityIndexes,axis = 0)
   print('{:<42} {}'.format("non-entity paragraphs in the docx:", 
                                 nonentityParaFrame.shape[0]))
   output = (
      parsed_object_list, #:List[Tuple[int,Docx_Paragraph_and_Runs]]
      parsed_object_lookup, #:Dict[int,Docx_Paragraph_and_Runs] = dict(parsed_object_list)
      doc_para_count, #: int = int(parsed_to_dict['total_encountered_paragraphs']) #type: ignore
      char_counts, #: Counter = parsed_to_dict['char_counts'] #type: ignore
      rootFrame,#:pd.DataFrame
      rootsubpieceFrame, #:pd.DataFrame
      lemmaFrame, #:pd.DataFrame
      nonentityParaFrame, #pd.DataFrame
      cleanerOutcomesDf #pd.DataFrame
      )
   print(cleanerOutcomesDf.head())
   with open('feature_Frames_and_Indexes.pkl', 'wb') as file:
      pickle.dump(output, file)