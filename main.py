import nbformat

# Open the notebook
with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=5)  # Read in the current version

# Save it in a lower format version
with open("main_v4.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f, version=4)  # Downgrade to version 4