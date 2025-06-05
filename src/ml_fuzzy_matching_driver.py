import time
import pandas as pd

df = Alteryx.read("#1")

input_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"
output_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"

request_time = time.time()
input_path = input_dir + "/pretfidf_fuzzy_match_prodtest_" + str(int(request_time)) + ".json"
output_path = output_dir + "/posttfidf_fuzzy_match_prodtest_" + str(int(request_time)) + ".json"

df.to_json(input_path, orient="records")
print(df)

python_env_path = "C:\\Users\\sdosi\\.virtualenvs\\Python_Outsourcing-PMVepcMz\\Scripts\\python.exe"
script_path = "C:\\Users\\sdosi\\ISN Summer 2024 Internship\\Projects\\Prospective Client Analysis\\Python Outsourcing\\ml_fuzzy_matching.py"
n_process = 4
n_process2 = 4

command = f'"{python_env_path}" "{script_path}" --i "{input_path}" --o "{output_path}" --np {n_process} --np2 {n_process2}'
# !{command}

df = pd.read_json(output_path, orient="records")
# df2 = pd.read_json(output_path.split(".")[0] + "_2.json", orient="records")
