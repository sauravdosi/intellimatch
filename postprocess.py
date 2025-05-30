from ayx import Alteryx
import pandas as pd
import ast

df = Alteryx.read("#1")

isn_active_hcs = Alteryx.read("#2")

isn_prospects_df = Alteryx.read("#3")

print(len(df))

unmatched_df = df[df["Match Score"] < 85]
matches_df = df[df["Match Score"] >= 85]
prospect_matches_df = matches_df[matches_df["Matched from Database Table"] == "DMSalesforce.dim.SalesforceAccount"]
df = matches_df[matches_df["Matched from Database Table"] != "DMSalesforce.dim.SalesforceAccount"]
print(len(unmatched_df))
print(len(matches_df))
print(len(prospect_matches_df))
print(len(df))

def get_sets(record):
    if "{" in record:
        return list(ast.literal_eval(record))
    else:
        x = set()
        x.add(int(record))
        return list(x)

df["Matched Company ID_CI"] = df["Matched Company ID_CI"].apply(lambda x: get_sets(x))

df_exploded = df.explode("Matched Company ID_CI")
#.rename(columns={"Matched Company ID_CI": "Matched ID_CI"})

df_exploded["join_index"] = df_exploded.index

merged_df = pd.merge(df_exploded, isn_active_hcs, left_on="Matched Company ID_CI", right_on="ID_CI")

merged_df["join_index"]
result = merged_df.groupby(merged_df.join_index)["CompanyName"].agg(list).reset_index()

result

result.columns = ["index", "CompanyNames"]
result
# df

final_df = pd.merge(df.reset_index(), result, on="index", how="left").drop(columns="index")

final_df["Match Type"] = final_df.apply(lambda x: "Current Name" if x["Matched Company Name"] in x["CompanyNames"] else "Previous Name", axis=1)
final_df = final_df.rename(columns={"CompanyNames": "ISN Database HC Match"})

final_df["ISN Database HC Match"] = final_df["ISN Database HC Match"].apply(lambda x: ", ".join(x))

final_df["Matched Company ISN ID"] = final_df["Matched Company ISN ID"].apply(lambda x: ", ".join(ast.literal_eval(x)) if "[" in x else x)

final_df


# Prospects Workflow

def get_sets_prospects(record):
    if "{" in record:
        return list(ast.literal_eval(record))
    else:
        x = set()
        x.add(record)
        return list(x)

prospect_matches_df["Matched Company ISN ID"] = prospect_matches_df["Matched Company ISN ID"].apply(lambda x: get_sets(x))

prospect_matches_df_exploded = prospect_matches_df.explode("Matched Company ISN ID")
#.rename(columns={"Matched Company ID_CI": "Matched ID_CI"})

prospect_matches_df_exploded["join_index"] = prospect_matches_df_exploded.index

prospects_merged_df = pd.merge(prospect_matches_df_exploded, isn_prospects_df, left_on="Matched Company ISN ID", right_on="ISN_Company_ID")

prospects_merged_df["join_index"]
prospects_result = prospects_merged_df.groupby(prospects_merged_df.join_index)["AccountName"].agg(list).reset_index()

prospects_result

prospects_result.columns = ["index", "AccountNames"]
prospects_result
# df

prospects_final_df = pd.merge(prospect_matches_df.reset_index(), prospects_result, on="index", how="left").drop(columns="index")

prospects_final_df["Match Type"] = prospects_final_df.apply(lambda x: "Current Name" if x["Matched Company Name"] in x["AccountNames"] else "Previous Name", axis=1)
prospects_final_df = prospects_final_df.rename(columns={"AccountNames": "ISN Database HC Match"})

prospects_final_df["ISN Database HC Match"] = prospects_final_df["ISN Database HC Match"].apply(lambda x: ", ".join(x))

prospects_final_df["Matched Company ISN ID"] = prospects_final_df["Matched Company ISN ID"].apply(lambda x: ", ".join(x) if type(x) == list else x)

# print(prospects_final_df["Matched Company ISN ID"].apply(lambda x: type(x)).unique())


# prospect_matches_df["ISN Database HC Match"] = prospect_matches_df["Matched Company Name"]
# prospect_matches_df["Matched Company Name"] = ""

final_matches_df = pd.concat([final_df, prospects_final_df])

final_matches_df


# # final_result.to_excel("C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/matched_historical_test.xlsx", sheet_name="Sheet1")
# master_count_df = final_matches_df["ISN Database HC Match"].value_counts().reset_index()
# master_count_df.columns = ["ISN Database HC Match", "Count"]

# master_count_df

# final_matches_df = pd.merge(final_matches_df, master_count_df, on="ISN Database HC Match")

# final_matches_df


unmatched_df["ISN Database HC Match"] = ""
unmatched_df["Matched Company Name"] = ""

final_result = pd.concat([final_matches_df, unmatched_df])

# final_result["AHC Company Name Upper"] = final_result["AHC Company Name"].apply(lambda x: x.upper().strip())

# final_result_count_df = final_result["AHC Company Name Upper"].value_counts().reset_index()
# final_result_count_df.columns = ["AHC Company Name Upper", "Count"]

# final_result = pd.merge(final_result, final_result_count_df, on="AHC Company Name Upper")

# final_result = final_result.drop_duplicates(subset=["AHC Company Name Upper"])
# final_result.drop(columns=["AHC Company Name Upper"], inplace=True)

final_result
print(final_result.columns)

# # result = df.groupby("Matched Company Name").agg(Count=("Matched Company Name", "size"), UniqueEntries=("AHC Company Name", lambda x: list(x.unique()))).reset_index()
# # final_result = final_result.rename(columns={"Count": "Total ISN Subscriber HC Count"})

# # count_df = final_result.groupby(["ISN Subscriber HC Company(s) (Best Match)", "AHC Company Name"]).size().reset_index(name="Count")
# # final_result = pd.merge(final_result, count_df, on=["ISN Subscriber HC Company(s) (Best Match)", "AHC Company Name"])
# # final_result = final_result.drop_duplicates(subset=["ISN Subscriber HC Company(s) (Best Match)", "AHC Company Name"])
final_result = final_result.sort_values(by=["Count", "ISN Database HC Match", "AHC Company Name"],
                           ascending=[False, True, True])

# # final_result = final_result.rename(columns={"Count": "ISN HC and AHC Pair Count"})

final_result = final_result[["AHC Company Name", "Matched Company Name", "ISN Database HC Match", "Matched Company ID_CI", "Matched Company ISN ID", "Match Score", "Match Type", "Match Category", "Report Column", "Count", "Source", "ID_CI", "CompanyID", "Matched from Database Table", "Matched Demo Prospect Client", "Matched Company Labels Dict", "labels_dict"]]

# final_result.to_excel("C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prefinal_agg_test.xlsx", sheet_name="Sheet1")
Alteryx.write(final_result, 1)