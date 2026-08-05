"""
data_loader.py

Loads any dataset CSV for the  project, checking its
column headings against the  dictionary (data_import_template.csv)
before loading it.

Two functions:
  - load_field_dictionary()  reads the master field dictionary
  - load_csv()                loads a real dataset, checking its headings first
"""

import csv


def load_field_dictionary(dictionary_path="data_import_template.csv"):
    """Reads the master field dictionary into a list of dictionaries."""
    with open(dictionary_path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_csv(data_path, dictionary_path="data_import_template.csv"):
    """
    Loads a real dataset CSV and checks every column heading:
      1. Is it lowercase with underscores (no spaces, no capitals)?
      2. Is it actually listed in the field dictionary?
    Raises an error naming the exact heading if either check fails.
    Returns the data as a list of dictionaries, one per row.
    """
    registered_names = [row["field_name"] for row in load_field_dictionary(dictionary_path)]

    with open(data_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for heading in reader.fieldnames:
            if heading != heading.lower() or " " in heading:
                raise ValueError(f"'{heading}' is not lowercase/underscore format")
            if heading not in registered_names:
                raise ValueError(f"'{heading}' is not in {dictionary_path}")

        data_rows = list(reader)

    print(f"Loaded {len(data_rows)} rows from {data_path}")
    return data_rows
