import click

def get_lm_line(lem_line, wf_line, has_msd, has_lemma):
    (lemma, msd), wf = lem_line.split(" # "), wf_line
    return (lemma if has_lemma else wf) + (f" # {msd}" if has_msd else "")

def get_lan(identifier):
    return identifier[:identifier.find("_")]

@click.command()
@click.option("--path", required=True)
@click.option("--identifier", required=True)
@click.option("--msd/--no-msd", required=True)
@click.option("--lemma/--wordform", required=True)
def main(path,identifier, msd, lemma):
    for split in "train dev tst remainder".split():
        infix = get_lan(identifier) if split == "remainder" else identifier
        lem_fn = f"{path}/{split}.{infix}.input"
        wf_fn = f"{path}/{split}.{infix}.output"

        msd_str = "msd" if msd else "no_msd"
        lem_str = "lemma" if lemma else "wordform"
        lm_infix = ".".join([infix, msd_str, lem_str])
        outf = open(f"{path}/{split}.{lm_infix}.lm", "w")
        for lem_line, wf_line in zip(open(lem_fn), open(wf_fn)):
            lem_line = lem_line.strip()
            wf_line = wf_line.strip()
            print(get_lm_line(lem_line, wf_line, msd, lemma),
                  file = outf)
            
if __name__=="__main__":
    main()
