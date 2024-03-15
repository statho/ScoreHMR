import torch
from torch import nn
from score_hmr.utils.model_blocks import SinusoidalPosEmb, ResMLPBlock


PREDICTORS = {
    "prohmr": {"thetas_emb_dim": 2048, "betas_emb_dim": 2048},
    "pare": {"thetas_emb_dim": 3072, "betas_emb_dim": 1536},
}


class FC(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.MODEL.USE_BETAS
        img_feats = cfg.MODEL.DENOISING_MODEL.IMG_FEATS
        hidden_dim = cfg.MODEL.DENOISING_MODEL.HIDDEN_LAYER_DIM
        # diffusion dimensions
        self.thetas_dim = cfg.MODEL.DENOISING_MODEL.POSE_DIM
        betas_dim = cfg.MODEL.DENOISING_MODEL.SHAPE_DIM if self.use_betas else 0
        self.diffusion_dim = self.thetas_dim + betas_dim
        # image features
        self.thetas_emb_dim = PREDICTORS[img_feats]["thetas_emb_dim"]
        self.betas_emb_dim = PREDICTORS[img_feats]["betas_emb_dim"]
        self.split_img_emb = self.use_betas and img_feats == "pare"

        # SMPL thetas
        time_dim = self.thetas_dim * 4
        sinu_pos_emb = SinusoidalPosEmb(self.thetas_dim)
        fourier_dim = self.thetas_dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_mlp = nn.Linear(in_features=self.thetas_dim, out_features=self.thetas_dim)
        self.blocks = nn.ModuleList([])
        for _ in range(cfg.MODEL.DENOISING_MODEL.NUM_BLOCKS_POSE):
            self.blocks.append(
                ResMLPBlock(
                    input_dim=self.thetas_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim,
                    cond_emb_dim=self.thetas_emb_dim,
                )
            )
        self.final_mlp = nn.Linear(in_features=self.thetas_dim, out_features=self.thetas_dim)

        # SMPL betas (optionally)
        if self.use_betas:
            time_dim = betas_dim * 4
            sinu_pos_emb = SinusoidalPosEmb(betas_dim)
            fourier_dim = betas_dim
            self.time_mlp_betas = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.init_mlp_betas = nn.Linear(in_features=betas_dim, out_features=betas_dim)
            self.blocks_betas = nn.ModuleList([])
            for _ in range(cfg.MODEL.DENOISING_MODEL.NUM_BLOCKS_SHAPE):
                self.blocks_betas.append(
                    ResMLPBlock(
                        input_dim=betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim,
                        cond_emb_dim=self.betas_emb_dim,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=betas_dim, out_features=betas_dim)


    def forward(self, x: torch.Tensor, time: torch.Tensor, cond_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            time : Tensor of shape [B] containing timesteps.
            cond_emb : Tensor of shape [B, cond_emb_dim] containing the image features to condition the model.
        Returns:
            torch.Tensor : predicted noise with shape [B, P].
        """
        if self.use_betas:
            thetas = x[:, :-10]
            betas = x[:, -10:]
            if self.split_img_emb:
                thetas_emb = cond_emb[:, :3072]
                cam_shape_emb = cond_emb[:, 3072:]
        else:
            thetas = x

        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)
        for block in self.blocks:
            thetas = block(thetas, tt, thetas_emb if self.split_img_emb else cond_emb)
        thetas = self.final_mlp(thetas)

        if self.use_betas:
            betas = self.init_mlp_betas(betas)
            tt_betas = self.time_mlp_betas(time)
            for block in self.blocks_betas:
                betas = block(
                    betas, tt_betas, cam_shape_emb if self.split_img_emb else cond_emb
                )
            betas = self.final_mlp_betas(betas)

            thetas_betas = torch.cat((thetas, betas), dim=1)
            return thetas_betas
        else:
            return thetas
