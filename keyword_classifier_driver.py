import time
import pandas as pd

df = Alteryx.read("#1")

# df["notclosed"] = df["word_vecs"].apply(lambda x: "-0.0077" in x[:10])
# df[df["notclosed"] == True]["word_vecs"].tolist()[2]
# print(df.dtypes)
# df = pd.read_json("C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest/postnlppreprocessing_prodtest_1722391850.json", orient="records")
# df2["notclosed"] = df2["CompanyName"].apply(lambda x: "Cenovus" in x)
# df2[df2["notclosed"] == True]["word_vecs"].tolist()[2]

# df2["word_vecs"].apply(lambda x: print(type(x)))

input_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"
output_dir = "C:/Users/sdosi/ISN Summer 2024 Internship/Projects/Prospective Client Analysis/data/test_output/prodtest"

request_time = time.time()
input_path = input_dir + "/prekeywordclassifier_prodtest_" + str(int(request_time)) + ".json"
output_path = output_dir + "/postkeywordclassifier_prodtest_" + str(int(request_time)) + ".json"

df.to_json(input_path, orient="records")

python_env_path = "C:\\Users\\sdosi\\.virtualenvs\\Python_Outsourcing-PMVepcMz\\Scripts\\python.exe"
script_path = "C:\\Users\\sdosi\\ISN Summer 2024 Internship\\Projects\\Prospective Client Analysis\\Python Outsourcing\\keyword_classifier.py"
n_process = 1 # No need of multiprocessing
mode = "test"
model = "ml_matching9_model.weights.h5"
command = f'"{python_env_path}" "{script_path}" --i "{input_path}" --o "{output_path}" --np {n_process} --m "{mode}" --mn "{model}"'
# !{command}

df = pd.read_json(output_path, orient="records")
