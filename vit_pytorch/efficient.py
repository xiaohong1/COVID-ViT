import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head=256, dropout = 0., emb_dropout = 0.):
        super().__init__()
        print('super=',super().__init__())
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 3 
        patch_dim = channels * patch_size ** 3
        print('xg:n_patches,p-dim, mlp_dim,depth=',num_patches,patch_dim, mlp_dim, depth)
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        #print('dim, mlp_dim=', dim, mlp_dim)
        print('ViT3D:self-pos,patch2-embedding,cls_tokem,dropput=:',self.pos_embedding[0].shape, self.patch_to_embedding,self.cls_token[0].shape,self.dropout)

        print('ViT:dim,mlp_dim=',dim,mlp_dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#        print('transfer=', self.transformer)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.Dropout(dropout)
        )


    def forward(self, img, mask = None):
        p = self.patch_size

        #x = rearrange(img, 'b c (w p1) (h p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = rearrange(img, 'b (d p3) (w p1) (h p2) -> b (w h d) (p1 p2 p3)',p1 = p, p2 = p, p3 = p)
        print('p,x-shape, img-shape=',p, x.shape, img.shape)
        x = self.patch_to_embedding(x)
        
        print('xg: x2embedding= ',x.shape)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        #x = self.transformer(x, mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
