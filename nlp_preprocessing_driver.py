import time
import pandas as pd
import re

df = Alteryx.read("#1")

df1 = df[((df["Source"] == "ISN Database") & (df["HasOwnerRole"] == 1))][:100]
df2 = df[(df["Source"] == "Salesforce")][:100]

df = pd.concat([df1, df2])

input_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"
output_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"

request_time = time.time()
input_path = input_dir + "/prenlppreprocessing_prodtest_" + str(int(request_time)) + ".json"
output_path = output_dir + "/postnlppreprocessing_prodtest_" + str(int(request_time)) + ".json"

df.to_json(input_path, orient="records")

python_env_path = "C:\\Users\\sdosi\\.virtualenvs\\Python_Outsourcing-PMVepcMz\\Scripts\\python.exe"
script_path = "C:\\Users\\sdosi\\ISN Summer 2024 Internship\\Projects\\Prospective Client Analysis\\Python Outsourcing\\nlp_preprocessing.py"
n_process = (len(df) // 200) + 1
# print(n_process)
command = f'"{python_env_path}" "{script_path}" --i "{input_path}" --o "{output_path}" --np {n_process}'
# !{command}

df = pd.read_json(output_path, orient="records")

# Alteryx.write(df, 1)