from os import listdir
from os.path import isfile, join

import stylecloud

data_path = "data/processed/keywords/"
fs = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for f in fs:
    fname = f[:-3] + "png"
    stylecloud.gen_stylecloud(
        file_path=join(data_path, f),
        stopwords=True,
        background_color='#1A1A1A',
        max_words=50,
        icon_name="fas fa-record-vinyl",
        output_name=f"vizs/decades/{fname}"
    )
