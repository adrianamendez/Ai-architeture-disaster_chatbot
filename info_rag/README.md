# Data Folder

This folder is **gitignored** — its contents are too large for GitHub (~4.7 GB total)
and have their own licensing on Kaggle. Re-create the structure locally before
running the notebook by downloading from Kaggle:

```
info_rag/
├── DISASTERS/                                                 (gitignored)
│   ├── 1900_2021_DISASTERS.xlsx - emdat data.csv
│   └── 1970-2021_DISASTERS.xlsx - emdat data.csv
├── global_disaster_response_2018_2024 (1).csv                 (gitignored)
└── images/                                                     (gitignored)
    ├── disaster_dataset/disaster_dataset/
    │   ├── fire/        *.jpg
    │   ├── flood/       *.jpg
    │   ├── landslide/   *.jpg
    │   ├── smoke/       *.jpg
    │   └── normal/      *.jpg
    └── earthquake/earthquake/
        └── *.jpg
```

## Sources

| What | Where to get it | Approx size |
|---|---|---|
| `global_disaster_response_2018_2024 (1).csv` | [Kaggle: Global Disaster Response Analysis (2018-2024)](https://www.kaggle.com/datasets/zubairdhuddi/global-daset) | ~4 MB |
| `DISASTERS/*.csv` | [Kaggle: All Natural Disasters 1900-2021 (EOSDIS)](https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis) | ~9 MB |
| `images/disaster_dataset/...` and `images/earthquake/...` | [Kaggle: Disaster Damage 5-Class Image Set](https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class) | ~4.7 GB |

## After downloading

1. Place files in the structure shown above.
2. From the project root, run the notebook (`disaster_chatbot.ipynb`) — the first
   `index_disaster_data()` call rebuilds `vector_db/` from the CSVs (~30 s on a laptop).
