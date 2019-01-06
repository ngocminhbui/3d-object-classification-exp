# 3D object classification code
Base on multiview method (http://vis-www.cs.umass.edu/mvcnn/).

## feat_extract:
extract 2048-d dimensional feature for each view (pretrained resnet50)

## ring classifier:
extract each ring (square ríng, circular rings) contains multiple views with different fusion stratergy: Fully connected, LSTM or average weighting.

## score net:
inferences attention-like score for a ring, produces the possibility of a ring being correctly classified by ring classifier.