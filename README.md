# Cross-Table Ablation of Graph-Augmented Table Retrieval

Code for a controlled study of where relational structure helps retrieval in
table question answering. The pipeline combines a text retriever, a GraphSAGE
table-graph retriever, score fusion, and a cross-encoder reranker, and isolates
the contribution of (i) inter-table foreign-key edges in the graph and (ii)
structure-aware reranking, using the multi-table MMQA benchmark.

## Layout

```
src/
  data/mmqa_loader.py          # MMQA loader: corpus, splits, inter-table edges
  graph/mmqa_graph.py          # multi-table corpus graph (intra / inter edges)
  graph/graph_embedding.py     # GraphSAGE table encoder
  evaluation/multitable_metrics.py  # full-set Recall@k, per-table Recall@k, MRR
scripts/
  inspect_mmqa.py              # quick dataset inspection
  build_mmqa_corpus.py         # build + cache the 702-table corpus and splits
  diagnose_mmqa_edges.py       # edge-extraction coverage diagnostics
  build_mmqa_graph.py          # build / validate the corpus graphs
  train_mmqa_graph.py          # train the graph leg (intra or inter)
  finetune_bge_mmqa.py         # in-domain fine-tuning of the text leg
  eval_mmqa_retrieval.py       # text-only / text+rerank baselines
  eval_mmqa_hybrid.py          # fusion, standard rerank, joint (FK-aware) rerank
  bootstrap_mmqa.py            # paired bootstrap + McNemar on full-set Recall@10
  mmqa_qualitative_example.py  # find rerank rescue examples
```

## Setup

```bash
pip install -r requirements.txt
```

The MMQA data is loaded from the public `table-benchmark/mmqa` dataset on the
Hugging Face Hub and is cached on first use; no manual download is needed.

## Reproducing the main results

```bash
# 1. Build the pooled corpus, splits, and inter-table edges
python scripts/build_mmqa_corpus.py

# 2. Build and sanity-check the corpus graphs (intra / inter)
python scripts/build_mmqa_graph.py

# 3. Train the graph leg (run both variants)
python scripts/train_mmqa_graph.py --variant intra --epochs 60
python scripts/train_mmqa_graph.py --variant inter --epochs 60

# 4. Fine-tune the text leg in domain
python scripts/finetune_bge_mmqa.py --epochs 5 --batch-size 32

# 5. Retrieval baselines
python scripts/eval_mmqa_retrieval.py --encoder mmqa --split test

# 6. Fusion and reranking (select alpha on dev, evaluate on test)
python scripts/eval_mmqa_hybrid.py --graph-variant inter --split dev --sweep-alpha
python scripts/eval_mmqa_hybrid.py --graph-variant inter --alpha 0.7 --split test
python scripts/eval_mmqa_hybrid.py --graph-variant inter --alpha 0.3 --split test --rerank
python scripts/eval_mmqa_hybrid.py --graph-variant inter --alpha 0.3 --split test --joint-rerank

# 7. Significance tests
python scripts/bootstrap_mmqa.py <run_a> <run_b> --label-a A --label-b B
```

Results are written under `data/results/`. Metrics are computed at a single
seed; the graph leg trains in a couple of minutes on CPU, while the text-leg
fine-tuning benefits from a GPU.

## Notes

- Inter-table edges are extracted from the gold SQL join conditions and the
  declared primary/foreign keys, with a value-overlap fallback for the small
  fraction of questions that neither source covers.
- The graph leg reads out a per-table node embedding from a single corpus-wide
  graph, so message passing can propagate across foreign-key-linked tables.
