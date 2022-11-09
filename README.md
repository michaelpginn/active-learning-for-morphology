## 1. Generate language model data

This script will generate data for the `lm.py` script. You can generate data either from lemmas or word forms (using the option `--lemma` or `--wordform`) and include MSDs or omit them (using the option `--msd` or `--no-msd`). 

If you run the script thus:

```
$ python3 generate_lm_files.py --path data/aym --identifier aym_S1 --msd --wordform
```

it will create four output files:
```
data/aym/dev.aym_S1.msd.wordform.lm
data/aym/train.aym_S1.msd.wordform.lm
data/aym/remainder.aym.msd.wordform.lm
data/aym/tst.aym_S1.msd.wordform.lm
```

The files `train.aym_S1.msd.wordform.lm` and `dev.aym_S1.msd.wordform.lm` are used to train the language model, the strings in file `remainder.aym.msd.wordform.lm` can be scored using the trained language model and `tst.aym_S1.msd.wordform.lm` is included for completeness.

## 2. Train LM

The training and development data can contain any combination of lemma/wordform and with/without MSD. Make sure to use the same configuration when running in test mode.

Input characters need to be separated by spaces and MSD features by semicolons. E.g.:

```
d o g # N;PL
```

```
$ python3 lm.py --mode train --train_file data/aym/train.aym_S1.msd.wordform.lm --dev_file data/aym/dev.aym_S1.msd.wordform.lm --model_file aym_S1.wordform.msd.pt --epochs 25
```

## 3. Score strings 

```
$ python3 lm.py --mode test --model_file aym_S1.wordform.msd.pt --test_file data/aym/remainder.aym.msd.wordform.lm
```

```
Output:

q u r p a n i # N;ACC;PL;PSS2S  1.2571778297424316
i n a m u k u # V;CF;PRS/PST+IMMED;3;SG;AC1+INCL        1.0466982126235962
j a m p ' a t i # V;CF;PRS/PST+IMMED;1;SG;AC3   0.9639571905136108
s u q ' u s a # N;PROPR 1.5546469688415527
l a q a t u # N;GEN;PL  1.5546469688415527
a n u   q a l l u # N;PRP;PL;PSS1S      0.9639571905136108
t u m a y k u # N;INTER;PSS3S   1.392483115196228
c h u q i l l a # N;TERM;PL;PSS1PL      1.1433143615722656
a n a t a # V;FUT;2;SG;AC1      1.392483115196228
```

