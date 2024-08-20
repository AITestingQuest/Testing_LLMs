import pandas as pd

df_tinyllama = pd.read_csv("test_result_tinyllama.csv", delimiter=";")
df_phi3 = pd.read_csv("test_result_phi3.csv", delimiter=";")
df_stablelm2 = pd.read_csv("test_result_stablelm2.csv", delimiter=";")

with open("test_result_summary.csv", 'w', encoding="utf-8") as file:
    file.write("LLM;PASS;FAIL\n")
    file.write(f"phi3;{df_phi3['Test Result'].value_counts().get(True, 0)};{df_phi3['Test Result'].value_counts().get(False, 0)}\n")
    file.write(f"stablelm2;{df_stablelm2['Test Result'].value_counts().get(True, 0)};{df_stablelm2['Test Result'].value_counts().get(False, 0)}\n")
    file.write(f"tinyllama;{df_tinyllama['Test Result'].value_counts().get(True, 0)};{df_tinyllama['Test Result'].value_counts().get(False, 0)}\n")




