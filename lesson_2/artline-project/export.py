import torchvision.transforms as T
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import load_learner
from pathlib import Path
from utils import *
from PIL import Image
import torch

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


def predict(img_url, learner, url=False):
    img = preprocess_image(img_url, url)
    _,pred, _ = learner.predict(img)
    pred = pred.detach().cpu().numpy().transpose((1, 2, 0))
    pred = (pred - pred.min())/ (pred.max() - pred.min()) * 255
    pred = pred.astype("uint8")
    pred = Image.fromarray(pred)
    return pred

img_url = "static/images/woman.jpg"
export = True
learner = load_learner(Path("."), 'checkpoint/ArtLine_920.pkl')
#prediction = predict(img_url, learner, url=False)
#visualize_image(prediction)
#img = preprocess_image(img_url, url=False)
if export:
    dummy_inp = torch.randn([1,3, 3091, 2227])
    torch.jit.save(torch.jit.trace(learner.model, dummy_inp), 'artline.pt')
print("")

