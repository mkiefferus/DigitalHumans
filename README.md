# Enhancing Text-to-Motion Synthesis through Language Models

## Setup
### Installing neccessary libraries
#### Text Enhancement
In order to run the text enhancement scripts and generate enhanced text descriptions, you will need to install the following python packages:
- spacy
- torch
- tqdm
- openai

Furthermore, you will need to download the ```en_core_web_sm``` model:
```
python -m spacy download en_core_web_sm
```

### Integrating External Repositories
#### MoMask
Clone the [MoMask Repository](https://github.com/EricGuo5513/momask-codes) into the folder ```DigitalHumans/external_repos``` and follow their instruction to download datasets and models

### OpenAI API Setup
Follow the instructions given in _"Step 2 - Set up your API key for all projects (recommended)"_ in the [OpenAI API Documentation](https://platform.openai.com/docs/quickstart?context=python) to configure your OpenAI API access.

## Usage
### Motion Description Enhancement
To generate new motion descriptions for the test dataset of HumanML3D using GPT-3.5 Turbo, run the following:
```
.\prompt_enhancement_models\text_refinement.py --system_prompt extra_sentence.json --folder_name extra_sentence_1
```
```system_prompt``` specifies the name of the JSON file including your desired system prompt, whereas ```folder_name``` (optional, by default "altered_text_" + current timestamp) specifies the name of the output directory inside ```prompt_enhancement/altered_texts/```.

Note that this script requires the HumanML3D dataset to be present in ```external_repos\momask-codes\dataset```. Furthermore, it currently always concatenates the GPT-3.5 output to the original motion description. This can be easily changed by adapting the return statement of the ```improved_prompt``` function inside ```text_refinement.py```.


## Contributors
- Anne Marx
- Axel Wagner
- Max Kieffer
- Michael Siebenmann
