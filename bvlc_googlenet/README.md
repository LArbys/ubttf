# BVLC GoogLeNet

Implementation of GoogLeNet.  Written to take weights from caffe-trained BVLC GoogLeNet model. Also can take weights from a network fine-tuned to select neutrino events from MicroBooNE images.

![Network](https://github.com/LArbys/ubttf/blob/master/bvlc_googlenet/screenshots/tensorboard_network.png)

## Tests

### On quail2.jpg

From caffe:

    top  0  predicted class: quail  prob= 0.996084
    top  1  predicted class: partridge  prob= 0.0035871
    top  2  predicted class: ruffed grouse, partridge, Bonasa umbellus  prob= 0.000324771
    top  3  predicted class: prairie chicken, prairie grouse, prairie fowl  prob= 2.65217e-06
    top  4  predicted class: black grouse  prob= 4.96185e-07


From tensorflow:

    Top  0 : index= 85   quail  prob= 0.974324
    Top  1 : index= 86   partridge  prob= 0.0181258
    Top  2 : index= 82   ruffed grouse, partridge, Bonasa umbellus  prob= 0.00719344
    Top  3 : index= 15   robin, American robin, Turdus migratorius  prob= 0.000216857
    Top  4 : index= 83   prairie chicken, prairie grouse, prairie fowl  prob= 2.18266e-05
    
### On dog.png

From caffe:

    top  0  predicted class: Tibetan mastiff  prob= 0.9993
    top  1  predicted class: Tibetan terrier, chrysanthemum dog  prob= 0.000471201
    top  2  predicted class: Bernese mountain dog  prob= 9.36844e-05
    top  3  predicted class: Newfoundland, Newfoundland dog  prob= 6.25618e-05
    top  4  predicted class: Rottweiler  prob= 4.58144e-05

From tensorflow:

    Top  0 : index= 244   Tibetan mastiff  prob= 0.98744
    Top  1 : index= 256   Newfoundland, Newfoundland dog  prob= 0.012395
    Top  2 : index= 249   malamute, malemute, Alaskan malamute  prob= 3.53301e-05
    Top  3 : index= 200   Tibetan terrier, chrysanthemum dog  prob= 3.45525e-05
    Top  4 : index= 154   Pekinese, Pekingese, Peke  prob= 2.52849e-05
    
### On quail277.JPEG

From caffe:

    top  0  predicted class: quail  prob= 0.793905
    top  1  predicted class: robin, American robin, Turdus migratorius  prob= 0.0718401
    top  2  predicted class: bulbul  prob= 0.0679601
    top  3  predicted class: chickadee  prob= 0.0375449
    top  4  predicted class: water ouzel, dipper  prob= 0.00748986
    
From tensorflow:

    Top  0 : index= 85   quail  prob= 0.812452
    Top  1 : index= 18   magpie  prob= 0.139093
    Top  2 : index= 19   chickadee  prob= 0.0262894
    Top  3 : index= 17   jay  prob= 0.0155182
    Top  4 : index= 16   bulbul  prob= 0.00307126


## Training

Working example!  Fine-tuning network previously trained to separate UB event images.

![Acc and Loss](https://github.com/LArbys/ubttf/blob/master/bvlc_googlenet/screenshots/tensorboard_loss_acc.png)
