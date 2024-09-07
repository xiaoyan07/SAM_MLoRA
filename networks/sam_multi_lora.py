from typing import Type, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT, Attention, window_partition, window_unpartition
from segment_anything.modeling.common import MLPBlock, LayerNorm2d
from networks.mlora import MultiLoraLinear
nonlinearity = partial(F.relu, inplace=True)

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
        self.scale = scale
        # self.Depth_Adapter = Adapter(dim, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

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

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)

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
    freeze_all_except_adapter(encoder)

    encoder = MultiLoraLinear.convert_lora_linear(encoder, r=32, num_lora=3, lora_alpha=32, lora_dropout=0.1, merge_weights=False)
    # encoder_lora = lora_on_attention(encoder, r=16, lora_alpha=16, lora_dropout=0.1, merge_weights=False)
    decoder = LinkNetDecoder([prompt_embed_dim] * 4, num_classes=1)
    model = EncoderDecoder(encoder, decoder)

    return model

def build_sam_vit_b_adapter_linknet_multi_lora(checkpoint=None, image_size=1024):
    return build_sam_adapter_linknet(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size
    ), [2, 5, 8, 11]