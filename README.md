# Italian MWE Identification

Dataset and code for an evaluation of how reliably language models identify and isolate
multiword expressions (MWEs) in authentic Italian sentences.

The resource contains 755 Italian MWEs annotated by PARSEME macro-category, each attested in
five sentences drawn from the CORIS corpus (3,775 sentences in total), together with the raw
outputs of three language models, a manual assessment of each output on a 0–3 scale, and a
manual annotation of the inflectional state of every occurrence.

## Contents

```
data/
  dataset.xlsx                  3,775 sentences with their target expression
  mwe_frequency.csv             755 expressions with category and corpus frequency
  model_outputs_scored.xlsx     model outputs and manual scores, one sheet per model
  inflection_annotation.xlsx    inflectional state of each occurrence
run_models.py                   queries the three models and collects their outputs
analysis.py                     reproduces all tables and figures
output/                         generated results (created by analysis.py)
```

## Data

### `dataset.xlsx`

One row per sentence, 3,775 rows.

| column | description |
|---|---|
| `ID` | unique identifier of the sentence, used as the join key across all files |
| `MWE` | the target expression in its canonical form |
| `Sentence` | the sentence as retrieved from CORIS |

Each of the 755 expressions is represented by five sentences.

### `mwe_frequency.csv`

One row per expression, 755 rows. Column names are in Italian.

| column | description |
|---|---|
| `Espressione` | the expression, matching the `MWE` column of `dataset.xlsx` |
| `Categoria` | PARSEME macro-category: `VMWE` (346), `NMWE` (234), `AMWE` (162), `FMWE` (13) |
| `freq_assoluta` | absolute frequency in itTenTen |
| `freq_per_milione` | frequency per million tokens in itTenTen |
| `CQL` | the corpus query used to count the expression |

Frequencies were counted with lemmatised queries covering inflected and discontinuous
realisations, so each value is the sum of the surface variants of the expression rather than
the frequency of its canonical form alone.

### `model_outputs_scored.xlsx`

Three sheets, `llama`, `mistral` and `gemma`, with 3,775 rows each.

| column | description |
|---|---|
| `ID`, `MWE`, `Sentence` | as in `dataset.xlsx` |
| `model_output` | the model's answer, stored verbatim |
| `score` | manual assessment on a 0–3 scale |

This file has the same structure as the `model_outputs.xlsx` produced by `run_models.py`; the
only difference is that the `score` column is filled in.

### `inflection_annotation.xlsx`

One row per sentence, 3,775 rows. Column names are in Italian.

| column | description |
|---|---|
| `ID` | identifier of the sentence |
| `MWE`, `Frase` | the expression and the sentence |
| `stato` | inflectional state: `canonica`, `inflessa contigua`, `discontinua`, or `n/d` |
| `clitico` | notes on clitics or pronouns inside or before the expression |

`canonica` marks an occurrence in the citation form, `inflessa contigua` an occurrence whose
components are inflected but adjacent, and `discontinua` one whose components are separated by
intervening material. `n/d` marks the cases in which the expression is absent from the retrieved
sentence or occurs with a literal reading, and these are excluded from the inflectional
analysis. Of the verbal occurrences, 1,722 carry one of the three states.

## Scoring scale

Each model output was assessed manually against the expected expression:

| score | meaning |
|---|---|
| 3 | correct identification and explicit isolation of the full MWE span |
| 2 | correct identification of the expression, without precise isolation of the span |
| 1 | partial identification: some components recovered, the full expression not |
| 0 | no identification, or an irrelevant or incorrect output |

Minor morphological variation was not penalised, provided the semantic and pragmatic identity of
the expression was preserved. All scoring was carried out by a single native speaker of Italian.

## Scripts

### `run_models.py`

Reads `dataset.xlsx`, submits each sentence to the three models through a local Ollama server,
and writes `model_outputs.xlsx` with one sheet per model and an empty `score` column.

The prompt is the one used in the study, in Italian:

```
Individua e isola la Multiword Expression presente all'interno della seguente frase,
restituendola nel formato:
-- MWE: [espressione individuata]
Frase: {sentence}
```

The script saves a checkpoint every 25 responses in `checkpoints/` and skips the identifiers it
has already processed, so an interrupted run can be resumed by launching it again. A failed
request is recorded as `ERROR: ...` in `model_output` and does not stop the run.

This script reproduces the querying procedure used in the study; it is not the file that was
originally executed.

### `analysis.py`

Reads the four data files and writes `output/analysis_results.xlsx`, with one sheet per analysis,
together with `frequency.png`, `category.png` and `length.png`.

It computes the distribution of scores per model, the correlation between expression frequency
and identification score, identification rates by syntactic category, sentence-length statistics
and the correlation between sentence length and score, and, within the verbal category, mean
scores by inflectional state with a Kruskal–Wallis test across the three states.

The script stops with an explicit message if a model leaves any sentence unscored or if an
expression has no frequency entry.

## Requirements

Python 3.9 or later, with `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl` and, for
`run_models.py`, `requests`.

```
pip install pandas numpy scipy matplotlib openpyxl requests
```

`run_models.py` additionally requires a running Ollama server on `localhost:11434` with the three
models pulled. It needs no API key, since inference is local.

## Reproducing the analysis

```
python analysis.py
```

Both scripts expect the data files in `data/`. To keep everything in a single directory, set
`DATA_DIR` to `pathlib.Path(".")` at the top of each script.


## Source of the sentences

The sentences were retrieved from CORIS, a reference corpus of written Italian compiled at the
University of Bologna, and are distributed here for research purposes only.
