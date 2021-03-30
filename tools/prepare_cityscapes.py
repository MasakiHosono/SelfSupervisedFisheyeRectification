"""
This script reshape original cityscapes dataset into our format.

- Before
SelfSupervisedFiseyeRectification
└── data
    └── cityscapes
        └── leftImg8bit_trainvaltest
            └── leftImg8bit
                ├── train
                │   ├── aachen
                │   ├── bochum
                │   ...
                ├── val
                └── test

- After
SelfSupervisedFiseyeRectification
└── data
    └── cityscapes
        ├── train
        │   ├── aachen
        │   ├── bochum
        │   ...
        ├── val
        ├── test
        ├── train.lst
        ├── val.lst
        └── test.lst
"""
