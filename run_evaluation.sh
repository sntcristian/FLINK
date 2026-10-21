#!/bin/bash
# Runs evaluate.py on every pipeline configuration.
# Each run is stored in results/<configuration>/, the scores of all runs in results/summary.txt.

INPUT_DIR=data/aida_testc
RESULTS_DIR=results_2
MODEL=dmis-lab/UltraBERT-Embed-150M
LIMIT=20

# Without graph re-ranking
python evaluate.py --input-dir $INPUT_DIR --output-dir $RESULTS_DIR/notrf_nograph \
    --no-trf-rerank --no-graph-rerank --limit $LIMIT --candidates $RESULTS_DIR/candidates.json

python evaluate.py --input-dir $INPUT_DIR --output-dir $RESULTS_DIR/trf_nograph \
    --trf-rerank --trf-model $MODEL --no-graph-rerank --limit $LIMIT \
    --candidates $RESULTS_DIR/candidates.json

# With graph re-ranking
for strategy in wikidata wikidata+wikipedia; do
    for centrality in degree betweenness eigenvector; do

        python evaluate.py --input-dir $INPUT_DIR --output-dir $RESULTS_DIR/notrf_${strategy}_${centrality} \
            --no-trf-rerank --graph-rerank --strategy $strategy --centrality $centrality --limit $LIMIT --candidates $RESULTS_DIR/candidates.json

        python evaluate.py --input-dir $INPUT_DIR --output-dir $RESULTS_DIR/trf_${strategy}_${centrality} \
            --trf-rerank --trf-model $MODEL --graph-rerank --strategy $strategy --centrality $centrality --limit $LIMIT --candidates $RESULTS_DIR/candidates.json

    done
done

# Summary: NER and NER + EL scores of every run

grep -H "^NER" $RESULTS_DIR/*/report.txt > $RESULTS_DIR/summary.txt
cat $RESULTS_DIR/summary.txt
