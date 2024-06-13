# Enhancing Text-to-Motion Synthesis through Language Models

## Demo
Our approach can enhance output motion quality by adding contextually relevant text signals or by translating high-level motion descriptions to a set of low-level motion descriptions.
In the following, we show 3 examples for qualitative improvements of the generated motion and one example for a degradation, where the LLM omitted important information.
Note that we subsequently built a quality check stage to minimize the occurrence of such refinement failures.
<!-- |Ground Truth|Unrefined (MoMask)|Refined (Ours)|
|:-:|:-:|:-:|
|![Broadjump GT](./media/broadjump_GT.gif)|![Broadjump Unrefined](./media/broadjump_unrefined.gif)| ![Broadjump Refined](./media/broadjump_refined.gif)|
| a person performs a typical broadjump | a person performs a typical broadjump | The person bends their arms and crouches down preparing for a jump, then extend their arms back as they propel themselves forward with their legs. |
|![Pitch GT](./media/pitch_GT.gif)|![Pitch Unrefined](./media/pitch_unrefined.gif)| ![Pitch Refined](./media/pitch_refined.gif)|
| a figure winds up for the pitch | a figure winds up for the pitch |The figure pulls back their arms in preparation for throwing something. |
|![Golf GT](./media/golf_GT.gif)|![Golf Unrefined](./media/golf_unrefined.gif)| ![Golf Refined](./media/golf_refined.gif)|
| person is performing a golf motion | person is performing a golf motion | Imitating a golf swing, the person assumes a stance and clasps their hand together in a golf grip, leans forward to simulate a put motion that swings from left to right. |
|![Circle GT](./media/circle_GT.gif)|![Circle Unrefined](./media/circle_unrefined.gif)| ![Circle Refined](./media/circle_refined.gif)|
| a person slowly walks in a counter clockwise circle | a person slowly walks in a counter clockwise circle | The person's arms are relaxed and swing gently with each step. | -->

<style>
  .custom-table {
    width: 100%;
    border-collapse: collapse;
  }
  .custom-table th, .custom-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
    vertical-align: middle;
  }
  .custom-table p {
    max-width: 300px;
    margin: auto;
    word-wrap: break-word;
  }
</style>

<table class="custom-table">
  <tr>
    <th>Ground Truth</th>
    <th>Unrefined (MoMask)</th>
    <th>Refined (Ours)</th>
  </tr>
  <tr>
    <td>
      <img src="./media/broadjump_GT.gif" width="200" height="200" alt="Broadjump GT" />
      <p>a person performs a typical broadjump</p>
    </td>
    <td>
      <img src="./media/broadjump_unrefined.gif" width="200" height="200" alt="Broadjump Unrefined" />
      <p>a person performs a typical broadjump</p>
    </td>
    <td>
      <img src="./media/broadjump_refined.gif" width="200" height="200" alt="Broadjump Refined" />
      <p>The person bends their arms and crouches down preparing for a jump, then extend their arms back as they propel themselves forward with their legs.</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./media/pitch_GT.gif" width="200" height="200" alt="Pitch GT" />
      <p>a figure winds up for the pitch</p>
    </td>
    <td>
      <img src="./media/pitch_unrefined.gif" width="200" height="200" alt="Pitch Unrefined" />
      <p>a figure winds up for the pitch</p>
    </td>
    <td>
      <img src="./media/pitch_refined.gif" width="200" height="200" alt="Pitch Refined" />
      <p>The figure pulls back their arms in preparation for throwing something.</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./media/golf_GT.gif" width="200" height="200" alt="Golf GT" />
      <p>person is performing a golf motion</p>
    </td>
    <td>
      <img src="./media/golf_unrefined.gif" width="200" height="200" alt="Golf Unrefined" />
      <p>person is performing a golf motion</p>
    </td>
    <td>
      <img src="./media/golf_refined.gif" width="200" height="200" alt="Golf Refined" />
      <p>Imitating a golf swing, the person assumes a stance and clasps their hand together in a golf grip, leans forward to simulate a put motion that swings from left to right.</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./media/circle_GT.gif" width="200" height="200" alt="Circle GT" />
      <p>a person slowly walks in a counter clockwise circle</p>
    </td>
    <td>
      <img src="./media/circle_unrefined.gif" width="200" height="200" alt="Circle Unrefined" />
      <p>a person slowly walks in a counter clockwise circle</p>
    </td>
    <td>
      <img src="./media/circle_refined.gif" width="200" height="200" alt="Circle Refined" />
      <p>The person's arms are relaxed and swing gently with each step.</p>
    </td>
  </tr>
</table>


## :gear: Setup

<details>

### Clone Repo

This repo relies on submodules ([MoMask](https://github.com/EricGuo5513/momask-codes)). Pull the whole repo with
```
git clone --recurse-submodules https://github.com/mkiefferus/DigitalHumans
```
External repos are found in the folder `external_repos`

### Setup MoMask Repo
<details>

*Disclaimer*: this section is the original setup-section from [MoMask](https://github.com/EricGuo5513/momask-codes). Please follow the link for further details.

### 1. Conda Environment
```
conda env create -f environment.yml
conda activate momask
pip install git+https://github.com/openai/CLIP.git
```
We test our code on Python 3.7.13 and PyTorch 1.7.1

#### Alternative: Pip Installation
<details>
We provide an alternative pip installation in case you encounter difficulties setting up the conda environment.

```
pip install -r requirements.txt
```
We test this installation on Python 3.10

</details>

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1b3GnAbERH8jAoO5mdWgZhyxHB73n23sK?usp=drive_link) to download the models and evaluators mannually.

### 3. Get Data

Follow the original [MoMask](https://github.com/EricGuo5513/momask-codes) repo to assemble the HumanML3D dataset.

</details>


### Setup Environment
Make sure to properly setup a separate environment with the `requirements.txt` file.
In case of any problems, these are the most important packages:
- spacy
- torch
- tqdm
- openai

Furthermore, you will need to download the ```en_core_web_sm``` model:
```
python -m spacy download en_core_web_sm
```

### Setup API Token

This project relies on LLMs for text refinement. Accessing these LLMs is done via the OpenAI client. When working with local language models, skip this part. 

When working with OpenAI models (GPT3.5-turbo, GPT4o, ...):
Please create an OPENAI API Token and export it as a global variable to your system. ```OPENAI_API_KEY = ".."```
Follow the instructions given in _"Step 2 - Set up your API key for all projects (recommended)"_ in the [OpenAI API Documentation](https://platform.openai.com/docs/quickstart?context=python) to configure your OpenAI API access.

## Usage
### Motion Description Enhancement
<details>
To generate new motion descriptions for the test dataset of HumanML3D using GPT-3.5 Turbo, run the following:

```
.\prompt_enhancement_models\text_refinement.py --system_prompt extra_sentence.json --folder_name extra_sentence_1 -r
```
* `--folder_name` : (optional) Specifies output folder name (generated automatically if not given: "altered_text_" + current timestamp at `prompt_enhancement/altered_texts/`)
* `--system_prompt` : Name of JSON file with correct system prompt
* `--batch_size` : Batch size for text enhancement (default: 1) (-1 will treat each line in each samples as new input to the language model. This increases output quality but may hit hard request limits per day.)
* `--continue_previous` : Path to folder where refining should be continued (skips already refined samples)
* `--refine_all_samples` : Refine all samples (default: refine test samples only)
* `--early_stop` : Stop after n refined batches (for testing)
* `--from_config` : Parameters in config file will overwrite respective args counterparts
* `-r` : Replace generated samples with original ones if do not match quality expectation
* `-d` : Delete generated samples with original ones if do not match quality expectation (inferior priority to `-r`)


*(Outdated) Note that this script requires the HumanML3D dataset to be present in `external_repos\momask-codes\dataset`. Furthermore, it currently always concatenates the GPT-3.5 output to the original motion description. This can be easily changed by adapting the return statement of the `improved_prompt` function inside `text_refinement.py`.*

</details>

## Contributors
- Anne Marx
- Axel Wagner
- Max Kieffer
- Michael Siebenmann
