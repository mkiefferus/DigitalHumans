
## Function Descriptions

### quality_control.py

**has_good_quality**

No docstring provided.

**delete_failed_files**

```
Delete faulty files from dataset
```

**replace_failed_files**

```
Replace faulty files with originals from the org_data_folder
```

**save_faulty_names**

```
Save names of faulty files to a text file
```

**check_dataset_quality**

No docstring provided.


### similarity_search_openai.py

**process_text**

No docstring provided.

**improved_prompt**

```
Given a motion description, finds similar motion descriptions in HumanML3D's training dataset,
uses these as an additional input for a language model to create an enhanced prompt.
TODO: current problems:
    - currently hardcoded for 2 neigbours
    - mirrored prompts (e.g. left only swapped with right) not detected

@param openai_client:
@param nbrs: NearestNeighbors object with embeddings
@param index_to_text: dictionary mapping embedding index to original text
@param text: motion description to refine
@param model: embedding model to use, likely CLIP
@return: enhanced motion description
```

**text_to_embedding**

```
Given a text, returns the feature vector by embedding with model

@param text: Text to embed
@param model: Embedding model to use, likely CLIP
@return: normalized embedding vector consisting of 512 floats
```

**dataset_to_neighbours**

```
Given a text file specifying which files belong to the dataset split, return a fitted NearestNeighbours object
containing the embeddings of all motion descriptions in these files along with a dictionary mapping
embedding index to text.
@param info_file_name: Text file specifying which files belong to the dataset split
@param model: Embedding model to use, usually CLIP
@return: NearestNeighbors object built with embeddings, dictionary mapping embedding index to text
```

**text_enhancement**

```
Runs full text enhancement pipeline for all files specified by info_file_name and saves them at folder
altered_texts.
@param openai_client:
@param info_file_name: Text file specifying which files belong to the dataset split
@param nbrs: NearestNeighbors object built with embeddings
@param index_to_text: Dictionary mapping embedding index to text
@param model: Embedding model to use, usually CLIP
```

**main**

No docstring provided.


### text_refinement.py

**_continue_folder**

```
(Helperfunction) Continue refining text at checkpoint
```

**_annotate_motion**

```
Add part of speech tags to motion description.
```

**export_data**

```
Generate dataset with refined text
```

**process_data**

```
Process files into JSON format and extract annotations.

Args:
    filenames (List[str]): List of filenames to be processed.
    observation_instead_of_motion (bool): if True, the function will put out a dictionary with key "observationX" instead of "motionX". Default is False.

Returns:
    Tuple[str, Dict[str, Dict[str]]]: A tuple containing the JSON string of motions and a dictionary of annotations.
```

**get_text_refinement**

```
Use OpenAI API for text refinement.
```

**refine_text**

```
Refines text in datafolder using given model and system prompt
```

**main**

No docstring provided.