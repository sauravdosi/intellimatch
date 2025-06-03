from ayx import Alteryx
import time
import pandas as pd
import re

df = Alteryx.read("#1")


def split_on_slash(text):
    name = re.sub(r"\b(c\s*\/\s*o|contractor\s*\/\s*operator)\b", "", text, flags=re.IGNORECASE)
    parts = [part.strip() for part in name.split("/")]
    parts = [part for part in parts if part]
    return parts


def separate_company_names(record):
    text = record["CompanyName"]
    companies = [text]

    if record["Source"] == "ISN Database" and (record["HasOwnerRole"] == 1 or record["HasOwnerRole"] == True):
        companies = []

        if type(text) == str:
            text = text.strip() if type(text) == str else text

            if ")" in text and text[-1] == ")":
                open_bracket_idx = text.index("(")
                bracket_text = text[open_bracket_idx + 1: -1]
                non_bracket_text = text[0: open_bracket_idx].strip()
                companies.append(non_bracket_text)
                # print(non_bracket_text)
                # print(bracket_text)
                if "/" in bracket_text:
                    companies.extend(split_on_slash(bracket_text))
                else:
                    if "APAC" not in bracket_text and "EMEA" not in bracket_text and \
                            not re.match(r"^\d{3}-\d{6}$", bracket_text):
                        companies.extend([bracket_text])
                    else:
                        # print("YAY")
                        # print(text)
                        companies.remove(non_bracket_text)
                        companies.append(text)

            elif "/" in text:
                companies.extend(split_on_slash(text))

            else:
                companies.append(text)

            # print(companies)
            companies = [company.strip() for company in companies if type(company) == str and company != ""]
            # print(companies)
    return companies


df["CompanyNameOriginal"] = df["CompanyName"]
df["CompanyName"] = df.apply(lambda x: separate_company_names(x), axis=1)
df["isSeparated"] = df["CompanyName"].apply(lambda x: len(x) > 1)

df = df.explode("CompanyName")

input_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"
output_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"

request_time = time.time()
input_path = input_dir + "/pretfidf_prodtest_" + str(int(request_time)) + ".json"
output_path = output_dir + "/posttfidf_prodtest_" + str(int(request_time)) + ".json"

df.to_json(input_path, orient="records")

python_env_path = "C:\\Users\\sdosi\\.virtualenvs\\Python_Outsourcing-PMVepcMz\\Scripts\\python.exe"
script_path = "C:\\Users\\sdosi\\ISN Summer 2024 Internship\\Projects\\Prospective Client Analysis\\Python Outsourcing\\kfold_tfidf_generator.py"
n_process = (len(df[df["Source"] == "Salesforce"]) // 1000) + 1
print(df["Source"].unique())
command = f'"{python_env_path}" "{script_path}" --i "{input_path}" --o "{output_path}" --np {n_process}'
!{command}

df = pd.read_json(output_path, orient="records")

Alteryx.write(df, 1)