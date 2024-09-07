from typing import Type, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT, Attention, window_partition, window_unpartition
from segment_anything.modeling.common import MLPBlock, LayerNorm2d

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Adapter3(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            scale: float = 0.5,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.MLP_Adapter = Adapter3(dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.Space_Adapter = Adapter3(dim)  # with skip connection
        self.scale = scale
        # self.Depth_Adapter = Adapter(dim, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        # self.Space_weight_Adapter = nn.Parameter(6 * torch.ones(1, ))
        # self.MLP_weight_Adapter = nn.Parameter(6 * torch.ones(1, ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Window partition
        # x: torch.Size([2, 48, 48, 768])
        # print(self.window_size) 14,14,0; 14,14,0; 14,14,0; 14,14,0;
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]  # 48,48
            x, pad_hw = window_partition(x, self.window_size)
            # torch.Size([32, 14, 14, 768]), (56, 56)

        # Space_weight = self.Space_weight_Adapter.sigmoid()
        # MLP_weight = self.MLP_weight_Adapter.sigmoid()

        x = self.attn(x)
        x = self.Space_Adapter(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.MLP_Adapter(xn)

        return x


class Resize(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, None, self.scale_factor, self.mode, self.align_corners)


class AdaImageEncoderViT(ImageEncoderViT):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            out_indices=(2, 5, 8, 11),
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes
        )
        self.out_indices = out_indices

        new_blocks = []
        for blk in self.blocks:
            new_blocks.append(
                Block(
                    dim=blk.dim,
                    num_heads=blk.num_heads,
                    mlp_ratio=blk.mlp_ratio,
                    qkv_bias=blk.qkv_bias,
                    norm_layer=blk.norm_layer,
                    act_layer=blk.act_layer,
                    use_rel_pos=blk.use_rel_pos,
                    rel_pos_zero_init=blk.rel_pos_zero_init,
                    window_size=blk.window_size,
                    input_size=blk.input_size,
                )
            )
        del self.blocks
        self.blocks = nn.ModuleList(new_blocks)

        self.pyramid_Adapter = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                Resize(scale, mode='bilinear', align_corners=True),
            ) for scale in [4, 2, 1]
        )
        self.last_resizer = Resize(0.5, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # cnn_output = self.cnn_Adapter(x)

        # x: torch.Size([2, 3, 768, 768])
        x = self.patch_embed(x)
        # x: torch.Size([2, 48, 48, 768])
        if self.pos_embed is not None:
            # self.pos_embed: torch.Size([1, 48, 48, 768])
            x = x + self.pos_embed  # torch.Size([2, 48, 48, 768])

        out_feats = []
        # len(self.blocks): 12
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out_feats.append(x.permute(0, 3, 1, 2))

        last_feat = out_feats.pop()
        pyramid_feats = []

        for out_feat, ada in zip(out_feats, self.pyramid_Adapter):
            pyramid_feats.append(ada(out_feat))

        pyramid_feats.append(self.last_resizer(self.neck(last_feat)))

        return pyramid_feats


def freeze_all_except_adapter(module):
    for name, p in module.named_parameters():
        if 'Adapter' not in name:
            p.requires_grad = False

    trainable_param_names = []
    for name, p in module.named_parameters():
        if p.requires_grad:
            trainable_param_names.append(name)
    print(f'trainable params: {trainable_param_names}')


nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class LinkNetDecoder(nn.Module):
    def __init__(self, filters, num_classes=1):
        super().__init__()

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # print(x_rd.shape) torch.Size([2, 256, 128, 128])
        e1, e2, e3, e4 = x
        # cnn_1, cnn_2, cnn_3, cnn_4 = cnn_x
        # shape: torch.Size([2, 128, 256, 256]), torch.Size([2, 128, 128, 128]), torch.Size([2, 128, 64, 64]), torch.Size([2, 256, 32, 32])
        # Decoder
        # torch.Size([4, 32, 512, 512])
        # torch.Size([4, 64, 256, 256])
        # torch.Size([4, 128, 128, 128])
        # torch.Size([4, 256, 64, 64])

        d4 = self.decoder4(e4) + e3  # torch.Size([2, 256, 64, 64])
        d3 = self.decoder3(d4) + e2  # torch.Size([2, 256, 128, 128])
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        # print(x.shape) torch.Size([2, 3, 1024, 1024])
        feats = self.enc(x)
        y = self.dec(feats)
        # print(y.shape) torch.Size([2, 1, 1024, 1024])
        return y

def resize_pretrained_pos(pos_embed: torch.Tensor, size):
    if pos_embed.ndim == 4:
        pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2), size, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
    elif pos_embed.ndim == 2:
        pos_embed = F.interpolate(pos_embed.unsqueeze(0).permute(0, 2, 1), size, mode='linear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 1).squeeze(0)
    return pos_embed


def resize_model_pos_embed(model, img_size, encoder_global_attn_indexes):
    pos_embed = F.interpolate(model.pos_embed.permute(0, 3, 1, 2), img_size // 16, mode='bicubic', align_corners=False)
    model.pos_embed.data = pos_embed.permute(0, 2, 3, 1)

    for i in encoder_global_attn_indexes:
        blk = model.blocks[i]
        new_size = 2 * img_size // 16 - 1

        new_rel_pos_h = F.interpolate(blk.attn.rel_pos_h.unsqueeze(0).permute(0, 2, 1), new_size, mode='linear',
                                      align_corners=False).permute(0, 2, 1).squeeze(0)
        blk.attn.rel_pos_h.data = new_rel_pos_h

        new_rel_pos_w = F.interpolate(blk.attn.rel_pos_w.unsqueeze(0).permute(0, 2, 1), new_size, mode='linear',
                                      align_corners=False).permute(0, 2, 1).squeeze(0)
        blk.attn.rel_pos_w.data = new_rel_pos_w

    return model


def build_sam_adapter_linknet(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint,
        image_size=1024
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    encoder = AdaImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        out_indices=encoder_global_attn_indexes,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        state_dict = {k.replace('image_encoder.', ''): v for k, v in state_dict.items() if 'image_encoder' in k}
        if image_size != 1024:
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'pos' in k:
                    if k == 'pos_embed':
                        v = resize_pretrained_pos(v, (image_size // 16, image_size // 16))
                    else:
                        blk_idx = int(k.split('.')[1])
                        if blk_idx in encoder_global_attn_indexes:
                            v = resize_pretrained_pos(v, 2 * image_size // 16 - 1)
                        else:
                            v = resize_pretrained_pos(v, 2 * 14 - 1)

                new_state_dict[k] = v
            state_dict = new_state_dict
            # state_dict = {k: v for k, v in state_dict.items() if 'pos' not in k}

        keys = encoder.load_state_dict(state_dict, strict=False)
        print(f'missing keys: {keys.missing_keys}')
        # assert all(['Adapter' in k for k in keys.missing_keys])

    # print(prompt_embed_dim) 256

    decoder = LinkNetDecoder([prompt_embed_dim] * 4, num_classes=1)
    model = EncoderDecoder(encoder, decoder)
    freeze_all_except_adapter(model.enc)

    return model


def build_sam_vit_b_adapter_linknet(checkpoint=None, image_size=1024):
    return build_sam_adapter_linknet(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size
    ), [2, 5, 8, 11]



