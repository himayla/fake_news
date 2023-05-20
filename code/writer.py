def write_data(df, path, cols=None, tsv=False):
    if cols:
        df.to_csv(f"{path}.csv", index_label="ID", columns=cols)
    else:
        df.to_csv(f"{path}.csv", index_label="ID")

    if tsv:
        df.to_csv(f"{path}.tsv", sep="\t")
    #df.to_excel(f"{path}.csv")
