Understood. Here is the revised markdown with the example command on a single line for easy copy-pasting.

-----

````markdown
# How to Run `docx2onehotdb.py`

This document provides instructions on how to set up your environment and run the `docx2onehotdb.py` script to process `.docx` files and store their contents in a OneHotDB.

---

## 1. Prerequisites

Before running the script, ensure you have the following installed and configured:

### Python
Make sure you have **Python 3** installed on your system. 🐍

### Required Libraries
The script requires one of the following libraries to read `.docx` files. You can install one using pip:

**Option A (Recommended):**
```bash
pip install python-docx
````

**Option B:**

```bash
pip install docx2txt
```

### Project Modules

The script also depends on custom modules `onehotdb.py` and `zmongo.py`. Ensure these files are in the same directory as `docx2onehotdb.py` or are accessible via your system's `PYTHONPATH`.

### Environment Variables

The script connects to a MongoDB instance using the ZMongo backend, which is configured through environment variables. You **must set these** before running the script.

**Example for bash/zsh:**

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DATABASE_NAME="my_onehot_db"
```

Replace the values with your actual MongoDB connection string and desired database name.

-----

## 2\. Running the Script

Once the prerequisites are met, you can execute the script from your terminal.

### Basic Usage

The script takes the path to a `.docx` file as its main argument.

```bash
python docx2onehotdb.py "C:\Users\iriye\Downloads\TerraEquity-20250829T193132Z-1-001\TerraEquity\research\The Florida Bar v. Reed, 644 So.2d 1355 (Fla. 1994).docx"
```

### Command-Line Arguments

You can customize the script's behavior using the following optional arguments:

  * `--link-key <KEY>`: Specifies a **unique key** to identify the document's data in the database. If not provided, the filename (without the extension) is used.
  * `--collection <NAME>`: Sets the name of the MongoDB collection where the encoded sentence data will be stored. The default is `sentences`.
  * `--vocab-collection <NAME>`: Sets the name of the collection for the vocabulary. The default is `onehot_vocab`.
  * `--print-top <NUMBER>`: After processing, this will print the top `N` words from the vocabulary found in the document. The default is `20`.

### Example Command

Here is a full example of how to run the script with all arguments on a single line:

```bash
python docx2onehotdb.py "MyReport.docx" --link-key "report_2023_q4" --collection "technical_reports" --vocab-collection "tech_vocab" --print-top 10
```

This command will:

1.  Read the content from `MyReport.docx`.
2.  Store the data under the logical key `report_2023_q4`.
3.  Save the encoded vectors into the `technical_reports` collection.
4.  Use or update the `tech_vocab` collection for the vocabulary.
5.  Print the first 10 words from the vocabulary that were found in the document. ✅

<!-- end list -->

```bash
python restore_words.py my_doc_key --collection "sentences" --vocab-collection "onehot_vocab"
```