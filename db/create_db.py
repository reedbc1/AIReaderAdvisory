import sqlite3
import json
import pandas as pd


# remove materials information from data
def remove_materials():
  new_data = []
  for record in data:
    new_record = {}
    for k, v in record.items():
      if k == 'materials':
        continue
      new_record.update({k: v})
    new_data.append(new_record)

  with open("json_files/wr_enhanced_new", "w") as f:
    json.dump(new_data, f, indent=2)


# create dvd table
def create_dvd_table():
  con = sqlite3.connect("wr_dvds.db")
  cur = con.cursor()
  cur.execute("""
    CREATE TABLE
    dvds(id, title, publicationDate, author,                contributers, summary, subjects)
    """)


# load json into sqlite
def load_data():
  with open("json_files/wr_enhanced_new.json", "r") as f:
    data = json.load(f)

  # see link for help with placeholders...
  # https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders
