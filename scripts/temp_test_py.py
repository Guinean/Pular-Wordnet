from typing import Optional, Dict, List, Any, Union, Tuple
import pydantic
from pydantic import ValidationError, validator, root_validator, Field, constr
from pydantic_docx import Docx_Paragraph_and_Runs, read_docx #type:ignore
from pydantic_docx_processor import create_sized_dataframe, expand_dataframe #type:ignore
import re
import json
from itertools import compress, chain
from datetime import datetime
import pandas as pd
import numpy as np
from functools import partial

import logging
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_-_%H-%M-%S")
logger_filename = f"logs_and_outputs/{current_time}docxFileParse.log"

handler = logging.FileHandler(logger_filename, 'w', 'utf-8') 
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

# logging.setLogRecordFactory(factory)
logging.basicConfig(handlers=[handler], level=logging.DEBUG)
logger = logging.getLogger()

import pickle
with open('all_white_paras.pkl', 'rb') as file:
    # Call load method to deserialze
    output = pickle.load(file, encoding='utf-8')

frame, obs = output


print(obs[0])
print(obs[0].cleaner())
print(obs[0])


