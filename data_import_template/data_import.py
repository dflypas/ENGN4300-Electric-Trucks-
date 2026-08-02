"""
data_loader.py

Generic CSV import for the Dairy Drive Electric project.

There are two kinds of file involved:

1. data_import_template.csv
   The master field dictionary. It does NOT hold real data, it lists
   every field the team will use across the whole project
   (fleet, route, energy, cost, emissions, time), what it means, its
   unit

2. Real data files (e.g. fleet_data.csv, route_data.csv)
   Each dataset lives in its own separate file. Its column headings
   must match field_name values from the dictionary above.

Because this script checks against the dictionary rather than any one
fixed set of columns, it works for every dataset the team creates --
now or later -- without needing to be rewritten.
"""

import csv


def load_field_dictionary(dictionary_path):
    """
    Reads data_import_template.csv and returns it as a list of
    dictionaries, one dictionary per registered field.
    """
    with open(dictionary_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader)

    print(f"Loaded {len(fields)} registered fields from {dictionary_path}")
    return fields


def get_registered_field_names(field_dictionary):
    """
    Pulls out just the field_name values from the dictionary,
    as a simple list, so we can check against it quickly.
    """
    field_names = []
    for field in field_dictionary:
        field_names.append(field["field_name"])
    return field_names


def is_valid_heading_style(heading):
    """
    Checks one column heading follows the naming rules:
      - all lowercase
      - no spaces
      - only letters, numbers, and underscores
    """
    if heading != heading.lower():
        return False
    if " " in heading:
        return False

    allowed_characters = "abcdefghijklmnopqrstuvwxyz0123456789_"
    for character in heading:
        if character not in allowed_characters:
            return False

    return True


def check_headings(headings, field_dictionary):
    """
    Checks a data file's column headings two ways:
      1. Do they follow the naming style (lowercase, underscores)?
      2. Are they actually registered in the field dictionary?
    Raises an error listing any problems found.
    """
    registered_names = get_registered_field_names(field_dictionary)

    bad_style = []
    not_registered = []

    for heading in headings:
        if not is_valid_heading_style(heading):
            bad_style.append(heading)
        elif heading not in registered_names:
            not_registered.append(heading)

    if len(bad_style) > 0:
        raise ValueError(
            f"These headings do not follow the standard style "
            f"(lowercase, underscores, no spaces): {bad_style}"
        )

    if len(not_registered) > 0:
        raise ValueError(
            f"These headings are not in data_import_template.csv -- "
            f"add them to the dictionary first: {not_registered}"
        )


def load_csv(data_path, dictionary_path="data_import_template.csv"):
    """
    Loads any real dataset CSV, checking its headings against the
    master field dictionary first. Returns the data as a list of
    dictionaries, one dictionary per row.

    Example:
        fleet_data = load_csv("fleet_data.csv")
        print(fleet_data[0]["vehicle_id"])
    """
    field_dictionary = load_field_dictionary(dictionary_path)

    with open(data_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        check_headings(reader.fieldnames, field_dictionary)
        data_rows = list(reader)

    print(f"Loaded {len(data_rows)} rows from {data_path}")
    return data_rows


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fields = load_field_dictionary("data_import_template.csv")
    print(fields[0])