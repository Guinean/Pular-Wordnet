# ToDo

Foundations Need:

1. Lemmas
   * POS
2. Glosses
   * Senses split

Simplifications

1. ignore plurals

Augmentations:

1. Root Groups
2. References / Ptr
   * https://wordnet.princeton.edu/documentation/wnsearch3wn#:-:text=pointer%20symbol
   * some may help matching, some may just give more lemmas / forms
3. Noun class (hard impl match, but likely powerful)
   * likely help matching
   * is this Domain Ptr?
4. Possibly sources/regions may be a split along which noise can be reduced.


Explore:
* lexographer files
* Domains
* Ptr's
* verb frames


## Approaches

* pydantic as wrapper for parser
* dump to 2d-tuple. Convert these to DataFrame, and split by type
  * approach would allow DF with rows per paragraph, columns per run, and complementary DFs encoding the style features. Paragraph features can be series only
  * with these, selection of all leading bold is easy.
  * can convert categorical, aggregate by count (and have additional features like leading run, para indent, etc). Concatenating these would allow tSNE or UMAP. 
    * This should allow outliers to be visible. Can also aggregate unique combinations of features into a line plot, and encode size as frequency. Basically to cluster paragraphs and runs by features. Can then simplify those into finite named run styles.
    * Can also do with with networkX, which probably makes more sense. Just work with Nodes and attributes, and edges just internal to a paragraph. 
      * reduce dimensionality, expose exceptions
      * normalize structure (root on diff line than lemma)
      * seek to identify heirachical features, ordinal, or categorical
        * focus cleaning on those
        * once those are certain, new edges can be created
        * lookup routines like Altair for mapping/placing/aligning based on categorical/numerical/ordinal/nominal variables for psuedo autonomous parsing. Greedy aggregation down ordinal. Allow passing a priority list for dynamic restructuring, or addition of new features
    * PCA MIGHT work on the raw version
* allow export to html? Or MD? Allow visualization inside a local browser



## Remaining problems

* confidently identify POS, isolate simple lemmas and find how to handle plurals
  * will need to find a way to checking which lemma form is present in the gloss?
  * How does PWN encode plurals? It doesn't
    * Can try to capture those for stemming rules?
  * Use other forms and synonyms as PTR's
    * dont need them initially provided I can isolate one. Those can be added later