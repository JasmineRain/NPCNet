from scipy import stats
import os
import numpy as np
import pandas as pd


def list_results():
    dir = "./visualization"
    df = pd.DataFrame(columns=["ID", "UNet", "AUNet", "BASNet", "Deeplabv3+", "HRNet", "NewModel", "RefineNet", "UNet++"])
    results = os.listdir(dir)
    print(results)
    for res in results:
        if res not in ["UNet", "AUNet", "BASNet", "Deeplabv3+", "HRNet", "NewModel", "RefineNet", "UNet++"]:
            continue
        samples = os.listdir(os.path.join(dir, res))
        samples.sort()
        print(res)
        names = list(map(lambda x: x.split("_")[0] + "_" + x.split("_")[1], samples))
        scores = list(map(lambda x: float("0." + x.split(".")[1]), samples))
        print(scores)
        df['ID'] = names
        df[res] = scores
        df.to_csv("./scores.csv")


if __name__ == "__main__":
    list_results()
