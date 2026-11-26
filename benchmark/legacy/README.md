# Legacy scripts

Earlier experiment scripts, kept for history. **None of them is used for the paper's results**; the current pipeline is described in `../REPRODUCE.md`.

Known issues, so they are not reused by mistake:
- `crop_benchmark_v2.py`: its Optuna "RF-meta stacking" tunes hyperparameters on **test** accuracy, which leaks test data into model selection.
- `final_paper_benchmark.py`, `gen_*`, `make_notebook.py`, `model_benchmark.py` and the notebooks: an earlier protocol and figure set, superseded by `onnx_winner_benchmark.py`, `tuned_ensembles.py`, `accuracy_push.py`, `novelty_experiments.py` and `paper_figures.py`.
