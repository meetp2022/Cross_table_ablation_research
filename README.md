# Cross-Table Ablation of Graph-Augmented Table Retrieval

Code for a controlled study of where relational structure helps retrieval in
table question answering. The pipeline combines a text retriever, a GraphSAGE
table-graph retriever, score fusion, and a cross-encoder reranker, and isolates
the contribution of (i) inter-table foreign-key edges in the graph and (ii)
structure-aware reranking, using the multi-table MMQA benchmark.

The single-table experiments (WikiTableQuestions, FinQA, TAT-QA) are also
included for the full ablation reported in the paper.

## Layout

```
src/
  data/dataset_loader.py         # single-table loaders (WikiTQ, FinQA, TAT-QA)
  data/mmqa_loader.py            # MMQA loader: corpus, splits, inter-table edges
  graph/table_to_graph.py        # single-table graph construction (NetworkX)
  graph/feature_extraction.py    # node feature extraction (BGE + positional)
  graph/graph_embedding.py       # GraphSAGE table encoder
  graph/mmqa_graph.py            # multi-table corpus graph (intra / inter edges)
  graph/train.py                 # contrastive training loop for the graph leg
  evaluation/metrics.py          # single-table R@1, R@5, MRR, exact match
  evaluation/multitable_metrics.py  # full-set Recall@k, per-table Recall@k, MRR
  utils/config.py                # YAML config loader
  utils/logging.py               # loguru setup
  pipelines/shared/              # vector store, query encoder, retriever, reranker, LLM client
  pipelines/text_baseline/       # text-only pipeline
  pipelines/graph_augmented/     # graph-only pipeline
  pipelines/hybrid/              # late-fusion hybrid pipeline
  pipelines/image_baseline/      # image baseline (not used in the paper)
configs/
  base_config.yaml               # shared settings
  pipeline1_text.yaml            # generic BGE text config
  pipeline1_text_finetuned.yaml  # fine-tuned BGE config
  pipeline3_graph.yaml           # graph pipeline config
scripts/
  # --- Multi-table (MMQA) ---
  inspect_mmqa.py                # quick dataset inspection
  build_mmqa_corpus.py           # build + cache the 702-table corpus and splits
  diagnose_mmqa_edges.py         # edge-extraction coverage diagnostics
  build_mmqa_graph.py            # build / validate the corpus graphs
  train_mmqa_graph.py            # train the graph leg (intra or inter)
  finetune_bge_mmqa.py           # in-domain fine-tuning of the text leg
  eval_mmqa_retrieval.py         # text-only / text+rerank baselines
  eval_mmqa_hybrid.py            # fusion, standard rerank, joint (FK-aware) rerank
  bootstrap_mmqa.py              # paired bootstrap + McNemar on full-set Recall@10
  mmqa_qualitative_example.py    # find rerank rescue examples
  # --- Single-table (WikiTQ, FinQA, TAT-QA) ---
  eval_retrieval.py              # single-table retrieval: text, graph, hybrid
  eval_hybrid_finetuned.py       # hybrid (FT-BGE + GraphSAGE) + rerank
  bootstrap_ci.py                # paired bootstrap 95% CI on R@1, R@5, MRR
  per_query_diagnostic.py        # 2x2 contingency + McNemar on per-query R@1
```

## Setup

```bash
pip install -r requirements.txt
```

The MMQA data is loaded from the public `table-benchmark/mmqa` dataset on the
Hugging Face Hub and is cached on first use; no manual download is needed.
Single-table datasets (WikiTableQuestions, FinQA, TAT-QA) are likewise loaded
from the Hub via the `datasets` library.

## Reproducing the main results

### Multi-table (MMQA)

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

### Single-table (WikiTQ, FinQA, TAT-QA)

```bash
# Text+rerank ablation (alpha=1.0 zeroes the graph leg)
python scripts/eval_hybrid_finetuned.py --dataset wikitq --split validation --max-samples 1000 --alpha 1.0

# Hybrid+rerank (alpha=0.7)
python scripts/eval_hybrid_finetuned.py --dataset wikitq --split validation --max-samples 1000

# Bootstrap CI comparing two runs
python scripts/bootstrap_ci.py data/results/<run_a> data/results/<run_b> \
    --label-a "Text+Rerank" --label-b "Hybrid+Rerank"

# Per-query diagnostic (contingency table + McNemar)
python scripts/per_query_diagnostic.py data/results/<run_a> data/results/<run_b> \
    --label-a "Text+Rerank" --label-b "Hybrid+Rerank"
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
- The single-table scripts share the same pipeline, config, and evaluation code
  as the multi-table scripts.
