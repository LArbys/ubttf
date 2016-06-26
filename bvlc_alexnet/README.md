# Test bed to build an AlexNet TensorFlow model. 

Aim is to be able to initialize with the BVLC-trained imagenet weights. Also, visualizing with tensorboard.

## Test of implementation using pictures

### quail277.JPEG

    Top  0 : index= 85   quail  prob= 0.497764
    Top  1 : index= 20   water ouzel, dipper  prob= 0.175486
    Top  2 : index= 17   jay  prob= 0.151597
    Top  3 : index= 18   magpie  prob= 0.064116
    Top  4 : index= 16   bulbul  prob= 0.0315789

### dog.png

    Top  0 : index= 256   Newfoundland, Newfoundland dog  prob= 0.3896
    Top  1 : index= 205   flat-coated retriever  prob= 0.339828
    Top  2 : index= 244   Tibetan mastiff  prob= 0.246388
    Top  3 : index= 219   cocker spaniel, English cocker spaniel, cocker  prob= 0.00553562
    Top  4 : index= 160   Afghan hound, Afghan  prob= 0.00393591
  
At some point, should check these against caffe-implementation results.
