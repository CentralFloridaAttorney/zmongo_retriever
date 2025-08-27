# OneHotDB: Command Examples & Parameter Guide

Below are one-line `bash` invocations of `onehotdb_demo.py` showing different configurations. Each section explains what the flags do and why you’d use them. Across all examples, every available parameter is demonstrated.

> Assumes `onehotdb_demo.py`, `onehotdb.py`, `zmongo.py`, and `data_processing.py` are importable in the current environment and your Mongo connection is configured as your `ZMongo` expects.

---

## 1) Baseline word-mode run (lowercase + stopwords, moderate limits)

```bash
python onehotdb_demo.py --source-collection cases --text-field text --vocab-collection onehot_vocab --vocab-key cases:word --index-collection cases_index --onehot-collection cases_onehot --mode word --limit 5000 --batch-size 1000 --log-level INFO
```

**What this does**

* `--source-collection cases` / `--text-field text`: read text from `cases.text`.
* `--vocab-collection onehot_vocab` / `--vocab-key cases:word`: store (or load) a vocabulary entry under this key.
* `--index-collection cases_index` / `--onehot-collection cases_onehot`: write **index** encodings to `cases_index`; write **bitset one-hot** encodings to `cases_onehot`.
* `--mode word`: tokenize by words using the default pattern `[A-Za-z0-9']+`.
* `--limit 5000` / `--batch-size 1000`: process up to 5k docs, fetching in batches of 1k.
* `--log-level INFO`: standard logging verbosity.

*Defaults implicitly used here*: lowercasing enabled, stopwords removed, OOV token included, `min_freq=1`, `max_vocab_size=None`.

---

## 2) Force a fresh vocab with frequency cut-offs and cap the size

```bash
python onehotdb_demo.py --source-collection cases --text-field text --vocab-collection onehot_vocab --vocab-key cases:word:v2 --index-collection cases_index_v2 --onehot-collection cases_onehot_v2 --mode word --min-freq 3 --max-vocab-size 5000 --force-refit --limit 20000 --batch-size 2000 --log-level DEBUG
```

**What this does**

* `--min-freq 3`: only tokens appearing **≥3** times are kept (reduces vocab noise).
* `--max-vocab-size 5000`: cap vocabulary at 5k most frequent tokens (keeps vectors compact).
* `--force-refit`: ignore any stored vocab and rebuild from `cases.text` now.
* `--vocab-key cases:word:v2`: version your vocab explicitly.
* `--index-collection ..._v2` / `--onehot-collection ..._v2`: write outputs to versioned collections.
* Bigger `--limit` and `--batch-size` to learn from more data; `--log-level DEBUG` for detailed diagnostics.

---

## 3) Character-mode encoding with case preserved and no OOV token

```bash
python onehotdb_demo.py --source-collection notes --text-field body --vocab-collection onehot_vocab --vocab-key notes:char:case --index-collection notes_index_char --onehot-collection notes_onehot_char --mode char --no-lowercase --no-oov --limit 10000 --batch-size 500 --log-level WARNING
```

**What this does**

* `--mode char`: build a character vocabulary (space kept; newlines/tabs omitted internally).
* `--no-lowercase`: keep original casing; useful when case may carry meaning (IDs, acronyms).
* `--no-oov`: **no `<UNK>`**; unseen characters map to `-1` in index mode.
* Target collections set for char runs; moderate `limit`/`batch-size`; quieter logs.

---

## 4) Custom word pattern, stopwords off, and a separate vocab collection

```bash
python onehotdb_demo.py --source-collection articles --text-field content --vocab-collection vocab_alt --vocab-key articles:word:hyphen --index-collection articles_index --onehot-collection articles_onehot --mode word --token-pattern '[A-Za-z][A-Za-z0-9_-]+' --no-stopwords --min-freq 2 --limit 8000 --batch-size 800 --log-level DEBUG
```

**What this does**

* `--vocab-collection vocab_alt`: keep this vocabulary separate from others.
* `--token-pattern '[A-Za-z][A-Za-z0-9_-]+'`: include hyphen/underscore tokens (e.g., `state-of-the-art`, `foo_bar`).
* `--no-stopwords`: retain all words, even common ones (may be desirable for IR tasks).
* `--min-freq 2`: drop singletons to cut noise.
* Debug logging to inspect what gets tokenized/kept.

---

## 5) Small sample pass reusing an existing vocab (no refit), tiny limit

```bash
python onehotdb_demo.py --source-collection memos --text-field text --vocab-collection onehot_vocab --vocab-key memos:word --index-collection memos_index --onehot-collection memos_onehot --mode word --limit 100 --batch-size 50 --log-level INFO
```

**What this does**

* With **no** `--force-refit`, the script tries to **load** `memos:word` vocab first; if found, it **reuses** it and **does not** re-fit.
* Processes only 100 docs to generate index + bitset outputs for a quick spot-check.
* Good for applying a previously built vocabulary to new or incremental data.

---

## 6) Tight vocabulary for compact indices; keep defaults elsewhere

```bash
python onehotdb_demo.py --source-collection transcripts --text-field cleaned --vocab-collection onehot_vocab --vocab-key transcripts:word:compact --index-collection transcripts_index_compact --onehot-collection transcripts_onehot_compact --mode word --max-vocab-size 2000 --limit 15000 --batch-size 1500 --log-level INFO
```

**What this does**

* `--max-vocab-size 2000`: restrict to top 2k words; **index mode** then needs only `ceil(log2(2000))=11` bits/token (theoretical), improving storage efficiency for indices.
* Larger processing window; standard logging.

---

## Parameter Reference (quick effects)

* **Input selection**

  * `--source-collection NAME` / `--text-field PATH`: where to read raw text (dot-path supported).
* **Vocabulary storage & identity**

  * `--vocab-collection NAME`: Mongo collection holding the vocab document.
  * `--vocab-key KEY`: `_id` of the vocab; version with suffixes like `:v2`.
* **Output destinations**

  * `--index-collection NAME`: collection for **index** encodings.
  * `--onehot-collection NAME`: collection for **bitset one-hot** encodings.
* **Tokenization & normalization**

  * `--mode {word,char}`: word tokens vs. characters.
  * `--token-pattern REGEX` (word mode): customize what counts as a token.
  * `--no-lowercase`: keep original case (default is lowercase).
  * `--no-stopwords`: keep common words (default removes a small stoplist).
* **Vocab shaping**

  * `--min-freq N`: drop rare tokens (frequency < N).
  * `--max-vocab-size K`: cap vocabulary to K most frequent tokens.
  * `--no-oov`: do not include `<UNK>`; unseen tokens index to `-1`.
* **Operational controls**

  * `--force-refit`: rebuild vocab now, ignoring any stored vocab for the key.
  * `--limit N`: maximum documents to process for both fitting and encoding.
  * `--batch-size N`: paging size for reading source documents.
  * `--log-level LEVEL`: `DEBUG|INFO|WARNING|ERROR` for verbosity control.

## Process A Single Document by _id

Process exactly one record by its string `_id` (write **indices** to `cases_index` and **bitsets** to `cases_onehot`):

```bash
python onehotdb_demo.py --source-collection documents --text-field text --vocab-collection onehot_vocab --vocab-key cases:word --index-collection cases_index --onehot-collection cases_onehot --doc-id 68a8ec0fa92e08f64a817e7b --mode word --limit 1 --batch-size 1 --log-level INFO
```

* `--doc-id ...` tells the demo to fetch that document from `cases` using the **string `_id`** you pass.
* It extracts the **starting text** from `text` (use a dot-path like `details.body` if needed).
* It encodes twice (index + bitset) and saves to the two target collections with the **same `_id`**.

If a vocab already exists under `--vocab-key`, it’s reused; otherwise a minimal vocab is fit from the fetched doc (or use `--force-refit` to rebuild).


### Implementation detail: Each run performs two writes—**index** encodings to your specified index collection and **bitset one-hot** encodings to your specified one-hot collection—so you can compare storage/throughput and choose the best fit for your pipeline.

