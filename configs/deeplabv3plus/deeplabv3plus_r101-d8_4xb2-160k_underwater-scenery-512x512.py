_base_ = './deeplabv3plus_r50-d8_4xb2-160k_underwater-scenery-512x512.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
