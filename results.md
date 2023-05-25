**TRAIN:**

kaggle, 4000 (only real)
fake_real, 4593?
liar, 12.790?

Accuracy
                kaggle      fake_real    liar
BERT-BASE       1.0         0.9763      0.6175
DISTILBERT      1.0         0.971       0.6094
ELECTRA         1.0         0.952       -
ROBERTA         -           -           -


**PREDICT:**

BERT-BASE UNCASED predictions (date unknown, 6 hours)

Data: 
kaggle, 11594
fake_real, 1379
Data: liar, 3838
             kaggle  fake_real      liar
accuracy   0.999827   0.979695  0.618551
f1         0.999842   0.979351  0.678242
precision  0.999684   0.982249  0.634718
recall     1.000000   0.976471  0.728174