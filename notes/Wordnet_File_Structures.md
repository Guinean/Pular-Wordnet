# Wordnet File structures
https://wordnet.princeton.edu/documentation/wndb5wn

For each class Noun, Verb, Adj, Adv

    Index file
    Data file
    Exception File

## Index File Format

Each index file begins with several lines containing a copyright notice, version number, and license agreement. These lines all begin with two spaces and the line number so they do not interfere with the binary search algorithm that is used to look up entries in the index files. All other lines are in the following format. In the field descriptions, number always refers to a decimal integer unless otherwise defined.

lemma  pos  synset_cnt  p_cnt  [ ptr_symbol...]  sense_cnt  tagsense_cnt   synset_offset  [ synset_offset...]

Exampled: Baabaen  n  2  5?  [ " ".join(ptr_symbol)?]  2  1  4123

### **lemma**

    &&YES
    lower case ASCII text of word or collocation. Collocations are formed by joining individual words with an underscore (_ ) character.

### **pos**

    &&YES
    Syntactic category: n for noun files, v for verb files, a for adjective files, r for adverb files.

### All remaining fields are with respect to senses of lemma in pos

#### **synset_cnt**

    &&YES
    Number of synsets that lemma is in. This is the number of senses of the word in WordNet. See Sense Numbers below for a discussion of how sense numbers are assigned and the order of synset_offset s in the index files.

#### **p_cnt**

    &&TRY/LIMITED
    Number of different pointers that lemma has in all synsets containing it.

#### **ptr_symbol**

    &&TRY/LIMITED
    A space separated list of p_cnt different types of pointers that lemma has in all synsets containing it. See wninput(5WN) for a list of pointer_symbol s. If all senses of lemma have no pointers, this field is omitted and p_cnt is 0 .

#### **sense_cnt**

    &&YES
    Same as sense_cnt above. This is redundant, but the field was preserved for compatibility reasons.

#### **tagsense_cnt**

    &&NO
    Number of senses of lemma that are ranked according to their frequency of occurrence in semantic concordance texts.

#### **synset_offset**

    &&YES
    Byte offset in data.pos file of a synset containing lemma . Each synset_offset in the list corresponds to a different sense of lemma in WordNet. synset_offset is an 8 digit, zero-filled decimal integer that can be used with **fseek(3)** to read a synset from the data file. When passed to **read_synset(3WN)** along with the syntactic category, a data structure containing the parsed synset is returned.

## Data File Format

Each data file begins with several lines containing a copyright notice, version number, and license agreement. These lines all begin with two spaces and the line number. All other lines are in the following format. Integer fields are of fixed length and are zero-filled.

    synset_offset  lex_filenum  ss_type  w_cnt  word  lex_id  [ word  lex_id...]  p_cnt  [ ptr...]  [ frames...]  |   gloss

    Example: 

    4123  ??lex_filenum  n  40  Baabaen  
  
### **synset_offset**

    &&YES
    Current byte offset in the file represented as an 8 digit decimal integer.

### **lex_filenum**

    &&MAYBE?INHERIT?
    Two digit decimal integer corresponding to the lexicographer file name containing the synset. See lexnames(5WN) for the list of filenames and their corresponding numbers.

### **ss_type**

    &&YES
    One character code indicating the synset type:

    n    NOUN
    v    VERB
    a    ADJECTIVE
    s    ADJECTIVE SATELLITE
    r    ADVERB

### **w_cnt**

    &&YES
    Two digit hexadecimal integer indicating the number of words in the synset.

### **word**

    &&YES
    ASCII form of a word as entered in the synset by the lexicographer, with spaces replaced by underscore characters (_ ). The text of the word is case sensitive, in contrast to its form in the corresponding index. pos file, that contains only lower-case forms. In data.adj , a word is followed by a syntactic marker if one was specified in the lexicographer file. A syntactic marker is appended, in parentheses, onto word without any intervening spaces. See wninput(5WN) for a list of the syntactic markers for adjectives.

### **lex_id**

    &&MAYBE?INHERIT?
    One digit hexadecimal integer that, when appended onto lemma , uniquely identifies a sense within a lexicographer file. lex_id numbers usually start with 0 , and are incremented as additional senses of the word are added to the same file, although there is no requirement that the numbers be consecutive or begin with 0 . Note that a value of 0 is the default, and therefore is not present in lexicographer files.

### **p_cnt**

    &&MAYBE?INHERIT?/LIMITED
    Three digit decimal integer indicating the number of pointers from this synset to other synsets. If p_cnt is 000 the synset has no pointers.

### **ptr**

    &&MAYBE?INHERIT?/LIMITED
    A pointer from this synset to another. ptr is of the form:

        pointer_symbol  synset_offset  pos  source/target 

    where synset_offset is the byte offset of the target synset in the data file corresponding to pos .

    The source/target field distinguishes lexical and semantic pointers. It is a four byte field, containing two two-digit hexadecimal integers. The first two digits indicates the word number in the current (source) synset, the last two digits indicate the word number in the target synset. A value of 0000 means that pointer_symbol represents a semantic relation between the current (source) synset and the target synset indicated by synset_offset .

    A lexical relation between two words in different synsets is represented by non-zero values in the source and target word numbers. The first and last two bytes of this field indicate the word numbers in the source and target synsets, respectively, between which the relation holds. Word numbers are assigned to the word fields in a synset, from left to right, beginning with 1 .

    See wninput(5WN) for a list of pointer_symbol s, and semantic and lexical pointer classifications.

### **frames**

    &&NO
    In data.verb only, a list of numbers corresponding to the generic verb sentence frames for word s in the synset. frames is of the form:

        f_cnt   +   f_num  w_num  [ +   f_num  w_num...] 

    where f_cnt a two digit decimal integer indicating the number of generic frames listed, f_num is a two digit decimal integer frame number, and w_num is a two digit hexadecimal integer indicating the word in the synset that the frame applies to. As with pointers, if this number is 00 , f_num applies to all word s in the synset. If non-zero, it is applicable only to the word indicated. Word numbers are assigned as described for pointers. Each f_num  w_num pair is preceded by a + . See wninput(5WN) for the text of the generic sentence frames.

### **gloss**

    &&YES
    Each synset contains a gloss. A gloss is represented as a vertical bar (| ), followed by a text string that continues until the end of the line. The gloss may contain a definition, one or more example sentences, or both.