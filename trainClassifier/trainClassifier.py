
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../fhgClassifier/')

from SimpleCV import SVMClassifier
from circularHOGExtractor import circularHOGExtractor



    
ch = circularHOGExtractor(4,2,6) 

extractor = [ch] 
svm = SVMClassifier(extractor) # try an svm, default is an RBF kernel function
trainPaths = []
classes = []
trainPaths.append('./yes/')
trainPaths.append('./no/')
classes.append('yes')
classes.append('no')
        
# train the classifier on the data
svm.train(trainPaths,classes,verbose=True)
svm.save('svmWildebeest.xml')
