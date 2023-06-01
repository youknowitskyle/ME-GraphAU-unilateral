from model.tiny_vit import tiny_vit_21m_384
from model.ANFL import MEFARG

model = tiny_vit_21m_384(pretrained=True)
print(model)

net = MEFARG(num_classes=8, backbone="tiny_vit_21m_384",
                 neighbor_num=3, metric="dots")

print(net)
print(next(net.backbone.parameters()))