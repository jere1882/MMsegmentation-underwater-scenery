_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/underwater-scenery.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    #pretrained=None,
    decode_head=dict(num_classes=7),
    auxiliary_head=dict(num_classes=7))
