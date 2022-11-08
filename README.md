## 1. Train LM

```
$ python3 lm.py --mode train --path data --language ady --epochs 10 --model_file ady.model.pt
```

## 2. Score strings 

```
$ python3 lm.py --mode test --model_file ady.model.pt --test_file data/ady/resample.ady.input
```

```
Output:

л ъ э б ы ӏ у # N;ERG;PL;DEF    0.9889280796051025
Х ь а о # N;INS;SG;DEF  1.314743995666504
к ъ а з м а к ъ б з ы у # N;INS;PL;NDEF 0.6802042722702026
ж ъ э р ы м # N;ERG;SG;DEF      1.0803543329238892
ж ъ о т # N;INS;SG;DEF  1.314743995666504
п э г ъ у а н э # N;NOM;PL      0.9889280796051025
ш а м г ь э н # N;INS;SG;NDEF   0.9889280796051025
ч ъ у ж ъ # N;NDEF;SG   1.314743995666504
б э г ь э х ъ # N;INS;PL;NDEF   0.9889280796051025
ц у м п э # N;NOM;SG    1.314743995666504
б ж ь ы н ы ф ы ц # N;NOM;SG    0.9096539616584778
```
