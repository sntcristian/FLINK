# FLINK: Featherweight Text-to-Wikidata Linker

The code contained in this repository is a programming exercise for building a lightweight Entity Linking approach for Wikidata that does not employ a local Wikidata index but simply retrieves and re-ranks entities based on information coming from the Wikidata API.

The implementation contains 3 sub-modules: 

* a [Named Entity Recognition](src/ner.py) module that identifies mentions in English sentences using [Spacy](https://spacy.io/models/en#en_core_web_lg);
* a [Candidate Retrieval and Re-ranking](src/retriever.py) module that, for each mention, finds relevant candidates using the Wikidata API and re-ranks them based on semantic similarity using a Sentence-transformer model;
* a [Graph Re-ranking](src/graph_reranker.py) module that builds for each sentence a graph connecting the retrieved candidates based on existing triples in Wikidata, and re-ranks them based on centrality measures.

The evaluation shows that combining semantic re-ranking with [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) with graph re-ranking using degree centrality provides the best results on test data.

## Install Requirements

```bash
conda create -n FLINK -y python=3.11 && conda activate FLINK
pip install -r requirements.txt
```

## Use Pipeline for Inference

```bash
python run_pipeline.py "Washington crossed the Delaware River before the Battle of Trenton." \
    --no-trf-rerank --limit 50 
```

## Evaluation

In order to evaluate the impact of each re-ranking strategy, the pipeline was tested on the [aida_testc](data/aida_testc/) dataset [(Shavarani & Sarkar, EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.686/). 

You can run a specific configuration of the pipeline on this evaluation dataset using [evaluate.py](./evaluate.py).

```bash
python evaluate.py --input-dir data/aida_testc/ --output-dir results/trf_wikidata_degree \
    --trf-rerank --graph-rerank --strategy wikidata --centrality degree --limit 20
```

You can test all possible configurations using the Shell script [run_evaluation.sh](./run_evaluation.sh).

## TO DO:

* Add support for languages different than English;
* Replace Spacy NER model with a multilingual GliNER model;
* Integrate small instruction-tuned language model for supporting candidate retrieval with query expansion.