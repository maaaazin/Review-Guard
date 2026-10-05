# Dataset setup

ReviewGuard expects a labelled hotel-review dataset, such as the **Deceptive
Opinion Spam Corpus** (often distributed through Kaggle). Download the dataset
from a source you are permitted to use and place its CSV or TSV file directly
in this directory (or in a subdirectory).

The loader searches recursively for `.csv`, `.tsv`, and `.txt` files and tries
to infer review and label columns. Typical names are `review`, `text`,
`review_text`, `deceptive`, `label`, or `class`.

If inference is ambiguous, run training with explicit column names:

```bash
python -m src.train --data-file data/your_file.csv --text-column review --label-column deceptive
```

Valid target values include `truthful`/`deceptive`, `real`/`fake`, and `0`/`1`.
They are normalized to `0 = truthful` and `1 = deceptive`.
