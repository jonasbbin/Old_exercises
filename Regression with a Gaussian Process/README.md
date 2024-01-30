# Disclaimer:
This was an old exercise. Therefore it might be possible that current approaches largely differ from this approach. Also keep in mind that this task was solved under potential restrictions (solution approaches, model usage, hardware, time).
# Active Learning
The aim of this exercise was to use Gaussian Processes together with medical data, to predict a variable.
# Our Approach
We use a RationalQuadratic kernel for our Gaussian Process. We pick our features using the GenericUnivariateSelect tool from sklearn.