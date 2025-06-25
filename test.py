import pandas as pd
import numpy as np
import ast

# df = pd.read_json("data/tfidf_reference.json", orient="records")
# #
# # print(len(df))
# #
# # df.to_csv("data/ml_matching_train_data5.csv", index=False)
#
# # df = pd.read_csv("data/nlp_reference.csv")
#
# df["Source"] = "database"
#
# print(df["Source"])
#
# df.to_json("data/tfidf_reference.json", orient="records")

import pandas as pd

# load your two JSON files
df_main = pd.read_json('data/kw_train_nlp.json', orient="records")       # has your primary data
df_other = pd.read_json('data/ml_matching_train_data5.json', orient="records")     # has the column you want

# say the column you want from df_other is 'foo',
# and the join key is 'id' in both frames:
#
#  ┌────────┐   ┌─────────┐
#  │ df_main│   │ df_other│
#  │        │   │         │
#  │ id     │   │ id      │
#  │ value1 │   │ foo     │
#  │ value2 │   └─────────┘
#  └────────┘
df_main = df_main[df_main["Source"] == "inference"]
df_main = df_main.sort_values("CompanyName").reset_index(drop=True)
df_other = df_other.sort_values("CompanyName").reset_index(drop=True)
print(df_main)
print(df_other)
# extract only the key + the one column you need:
# now concatenate them column-wise:
df_out = pd.concat(
    [
        df_main,
        df_other['label_list']  # just the one column you need
    ],
    axis=1
)

# now df_out has all columns of df_main plus 'foo' from df_other
print(df_out)

df_out.to_json('data/new_train_data.json', orient="records")
