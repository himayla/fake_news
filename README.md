# Enhancing Online Fake News Classification With an Argumentation-Based Pipeline

Repository for thesis project for Information Studies: Data Science.

Abstract:

The proliferation of fake news is a growing concern affecting both individuals and society as a whole. The increasing amount of academic attention has led to the development of various promising methods for classifying fake news. However, the dynamic nature of fake news, where new texts of varying lengths and topics constantly emerge across different domains, presents ongoing challenges. This study attempts to address these challenges and contribute to the existing work in fake news detection by proposing a new argumentation-based pipeline. We investigate the performance of fine-tuning state-of-the-art pre-trained language models using the extracted argumentation from fake and real news texts as input. We obtain promising results, particularly for distinguishing fake and real news in long texts, with an overall accuracy of 0.83. Our findings suggest that leveraging argumentation knowledge in an argumentation-based pipeline is potentially valuable for improving fake news detection in the future.

## Structure

The Git repository is structured as follows:

* **code**: contains all the code for this project
  * **code/dolly**: contains the code to extract argumentation using [Dolly 2.0](https://huggingface.co/databricks/dolly-v2-3b)
    * `instruct_pipeline.py`: contains the file which prompts requests to Dolly 2.0
    * `extract.py`: contains the actual code
  * **code/margot**: contains all code to extract argumentation using [MARGOT](https://www.sciencedirect.com/science/article/pii/S0957417416304493)
    * `extract.py`: contains the code to run MARGOT
    * In this file the `predictor` needs to be placed, and all dependencies needs to be installed
  * `cleaner.py`: contains code that is used by `loader.py` to clean the text, either for the baseline or argumentation-based pipeline.
  * `loader.py`: contains code to load the original data and clean this, or the clean data
  * `params.json`: contains the parameters for training LLM
* `compile.sh`: contains a script to create necessary folders and loads and cleans the data
* `eda.ipynb`: contains the original Exploratory Data Analysis, performed prior to implementation of the rest of the code.
* `visualization.ipynb`: contains code for different visualizations of different experiments
