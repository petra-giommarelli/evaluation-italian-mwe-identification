import pathlib

import pandas as pd
import requests

DATASET_FILE = "dataset.xlsx"
OUTPUT_FILE = "model_outputs.xlsx"
CHECKPOINT_DIR = pathlib.Path("checkpoints")
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 600
SAVE_EVERY = 25

MODELS = {
    "llama": "llama3.3:70b",
    "mistral": "mistral:7b",
    "gemma": "gemma3:4b",
}

PROMPT_TEMPLATE = (
    "Individua e isola la Multiword Expression presente all'interno "
    "della seguente frase, restituendola nel formato:\n"
    "-- MWE: [espressione individuata]\n"
    "Frase: {sentence}"
)


def query_model(model_tag, sentence):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model_tag,
            "prompt": PROMPT_TEMPLATE.format(sentence=sentence),
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def load_checkpoint(path):
    if path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_model(name, model_tag, data):
    path = CHECKPOINT_DIR / f"{name}.csv"
    rows = load_checkpoint(path)
    done_ids = {row["ID"] for row in rows}
    total = len(data)

    for item in data.itertuples(index=False):
        if item.ID in done_ids:
            continue
        try:
            output = query_model(model_tag, item.Sentence)
        except Exception as error:
            output = f"ERROR: {error}"
        rows.append({"ID": item.ID, "model_output": output})
        if len(rows) % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{name}: {len(rows)}/{total}")

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"{name}: {len(rows)}/{total} complete")
    return pd.DataFrame(rows)


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    data = pd.read_excel(DATASET_FILE)[["ID", "MWE", "Sentence"]]

    results = {}
    for name, model_tag in MODELS.items():
        results[name] = run_model(name, model_tag, data)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for name, outputs in results.items():
            sheet = data.merge(outputs, on="ID", how="left")
            sheet["score"] = ""
            sheet.to_excel(writer, sheet_name=name, index=False)

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()