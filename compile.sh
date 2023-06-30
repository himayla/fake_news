pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords', quiet=True)"
python -c "import nltk; nltk.download('punkt', quiet=True)"
python -m spacy download en_core_web_sm

mkdir -p baseline/data; mkdir -p baseline/models

mkdir -p argumentation/data; mkdir -p argumentation/models

mkdir -p argumentation/data/dolly; mkdir -p argumentation/data/margot; 

mkdir -p argumentation/models/dolly/claims; mkdir -p argumentation/models/dolly/evidence; mkdir -p argumentation/models/dolly/structure
mkdir -p argumentation/models/margot/claims; mkdir -p argumentation/models/margot/evidence; mkdir -p argumentation/models/margot/structure

python code/loader.py data baseline/data
python code/loader.py data argumentation/data