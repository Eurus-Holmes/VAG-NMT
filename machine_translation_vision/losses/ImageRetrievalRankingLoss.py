import torch
from torch.autograd import Variable

class ImageRetrievalRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ImageRetrievalRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0

        return cost_s.sum()