# Datasets

Most datasets are loaded at runtime from `sklearn.datasets` and `seaborn`.
Two files are shipped locally so the course works offline:

| File | Used in | Source |
|------|---------|--------|
| `titanic.csv` | Ch12 Capstone | Kaggle "Titanic – Machine Learning from Disaster" train set (891 rows, public domain copy via datasciencedojo/datasets) |
| `../1-introduction/03-exercises/penguins.csv` | Ch01 bonus exercise | seaborn-data (Palmer Penguins, CC0) |

Load in notebooks with a path relative to the notebook folder, e.g.
`pd.read_csv('../../0-datasets/titanic.csv')`.

See `DIDAKTIK.md` for the full dataset inventory.
