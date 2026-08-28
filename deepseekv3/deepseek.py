import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelArgs:
    block_size = 512
    embeddings_dims = 512
    attn_dropout = 0.1
    no_of_heads = 8
    dropout = 0.1
    no_of_decoder_layers = 6
    vocab_size = 32000
    base_freq = 100000
    experts = 16
    top_experts = 4
    noisy_topk = False
    use_shared_expert = True
    useauxFreeLoadBalancingLoss = True
    aux_free_bias_update_rate = 0.001
    mtp_heads = 1
    latent_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"


class Normalization(nn.Module):
    def __init__(self, embeddings_dims=ModelArgs.embeddings_dims, eps=1e-6, device=ModelArgs.device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embeddings_dims, device=device))

    def forward(self, x):
        x_norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(x_norm + self.eps)
        return self.weight * x


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        swish = x * self.sig(x)
        return swish


class SWiGLUExpertMoE(nn.Module):
    def __init__(self, embeddings_dims=ModelArgs.embeddings_dims, device=ModelArgs.device):
        super().__init__()
        self.hidden_dims = ((embeddings_dims * 2) * 4) // 3
        self.swish = Swish()
        self.linear_layer1 = nn.Linear(embeddings_dims, self.hidden_dims, bias=False, device=device)
        self.linear_layer2 = nn.Linear(embeddings_dims, self.hidden_dims, bias=False, device=device)
        self.linear_layer3 = nn.Linear(self.hidden_dims, embeddings_dims, bias=False, device=device)

    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out


class MoeLayer(nn.Module):
    def __init__(
        self,
        embeddings_size=ModelArgs.embeddings_dims,
        experts=ModelArgs.experts,
        top_experts=ModelArgs.top_experts,
        noisy_topk=ModelArgs.noisy_topk,
        use_shared_expert=ModelArgs.use_shared_expert,
        useauxFreeLoadBalancingLoss=ModelArgs.useauxFreeLoadBalancingLoss,
        aux_free_bias_update_rate=ModelArgs.aux_free_bias_update_rate,
        device=ModelArgs.device,
    ):
        super().__init__()
        if top_experts > experts:
            raise ValueError("top_experts has to be <= experts")

        self.experts = experts
        self.top_experts = top_experts
        self.noisy_topk = noisy_topk
        self.use_shared_expert = use_shared_expert
        self.useauxFreeLoadBalancingLoss = useauxFreeLoadBalancingLoss
        self.device = device

        self.heads = nn.ModuleList(
            [SWiGLUExpertMoE(embeddings_dims=embeddings_size, device=device) for _ in range(experts)]
        )
        self.gate = nn.Linear(embeddings_size, experts, bias=False, device=device)

        if use_shared_expert:
            self.shared_expert = SWiGLUExpertMoE(embeddings_dims=embeddings_size, device=device)
        else:
            self.shared_expert = None

        if noisy_topk:
            self.noise = nn.Linear(embeddings_size, experts, bias=False, device=device)

        if useauxFreeLoadBalancingLoss:
            self.register_buffer("routing_bias", torch.zeros(experts, device=device))
            self.bias_update_speed = aux_free_bias_update_rate

    def forward(self, x):
        # x is [B,T,C]
        gate_out = self.gate(x)

        if self.noisy_topk and self.training:
            noise = self.noise(x)
            gaussian_noise = torch.randn_like(gate_out)
            gate_out = gate_out + F.softplus(noise) * gaussian_noise

        if self.useauxFreeLoadBalancingLoss:
            gate_out = gate_out + self.routing_bias

        top_k_values, top_k_indices = torch.topk(gate_out, k=self.top_experts, dim=-1)
        masked = torch.full_like(gate_out, torch.finfo(gate_out.dtype).min)
        masked_values = masked.scatter(-1, top_k_indices, top_k_values)
        probs = F.softmax(masked_values, dim=-1)

        flat_x = x.reshape(-1, x.size(-1))
        flat_probs = probs.reshape(-1, self.experts)
        flat_out = torch.zeros_like(flat_x)

        for i in range(self.experts):
            expert_i_is_chosen_mask = flat_probs[:, i] > 0
            if not expert_i_is_chosen_mask.any():
                continue

            selected_input_tokens = flat_x[expert_i_is_chosen_mask]
            active_token_weights = flat_probs[expert_i_is_chosen_mask, i]
            expert_out = self.heads[i](selected_input_tokens)
            flat_out[expert_i_is_chosen_mask] += expert_out * active_token_weights.unsqueeze(-1)

        out = flat_out.reshape_as(x)

        if self.shared_expert is not None:
            out = out + self.shared_expert(x)

        if self.useauxFreeLoadBalancingLoss and self.training:
            with torch.no_grad():
                ci = probs.sum(dim=(0, 1))
                ci_avg = ci.mean()
                error_i = ci_avg - ci
                update = self.bias_update_speed * torch.sign(error_i)
                self.routing_bias.add_(update)

        return out


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims=ModelArgs.embeddings_dims,
        block_size=ModelArgs.block_size,
        base_freq=ModelArgs.base_freq,
        device=ModelArgs.device,
    ):
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.block_size = block_size

        pe = torch.zeros(block_size, embeddings_dims, device=device)
        position = torch.arange(0, block_size, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embeddings_dims, 2, dtype=torch.float32, device=device)
            * (-math.log(base_freq) / embeddings_dims)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[-1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, start_pos=0):
        # x is [B,T,C]
        seq_len = x.shape[1]
        end_pos = start_pos + seq_len
        if end_pos > self.block_size:
            raise ValueError("sequence length is bigger than block_size")
        return self.pe[:, start_pos:end_pos]


class LatentAttention(nn.Module):
    def __init__(
        self,
        embeddings_dims=ModelArgs.embeddings_dims,
        no_of_heads=ModelArgs.no_of_heads,
        latent_dim=ModelArgs.latent_dim,
        attn_dropout=ModelArgs.attn_dropout,
        device=ModelArgs.device,
    ):
        super().__init__()
        if embeddings_dims % no_of_heads != 0:
            raise ValueError("embeddings_dims has to be divisible by no_of_heads")

        self.head_size= embeddings_dims // no_of_heads
        self.latent_dim= latent_dim
        self.W_k=nn.Linear(latent_dim, self.head_size, bias=False, device=device)
        self.W_v = nn.Linear(latent_dim, self.head_size, bias=False, device=device)
        self.W_dkv = nn.Linear(embeddings_dims, latent_dim, bias=False, device=device)
        self.query = nn.Linear(embeddings_dims, self.head_size, bias=False, device=device)
        self.dropout = nn.Dropout(p=attn_dropout)

    def _apply_masks(self, weights, mask, past_tokens):
        B, T, S = weights.shape
        mask_value = torch.finfo(weights.dtype).min

        causal_mask = torch.tril(
            torch.ones(T, S, dtype=torch.bool, device=weights.device),
            diagonal=past_tokens,
        )
        weights = weights.masked_fill(causal_mask.unsqueeze(0) == 0, mask_value)

        if mask is None:
            return weights

        if mask.ndim == 2:
            if mask.shape[1] < S:
                raise ValueError("2D mask needs at least key sequence length")
            key_mask = mask[:, :S].unsqueeze(1)
            weights = weights.masked_fill(key_mask == 0, mask_value)
        elif mask.ndim == 3:
            weights = weights.masked_fill(mask[:, :T, :S] == 0, mask_value)
        elif mask.ndim == 4:
            weights = weights.masked_fill(mask[:, 0, :T, :S] == 0, mask_value)
        else:
            raise ValueError("mask has to be 2D, 3D, or 4D")

        return weights

    def forward(self, x, kv_cache=None, mask=None):
        # x is [B,T,C], latent cache is [B,S,latent_dim]
        latent_matrix = self.W_dkv(x)
        past_tokens = 0

        if kv_cache is None:
            kv_cache = latent_matrix
        else:
            past_tokens = kv_cache.shape[1]
            kv_cache = torch.cat([kv_cache, latent_matrix], dim=1)

        absorbed_q = torch.matmul(self.query.weight.T, self.W_k.weight)
        q_res=torch.matmul(x, absorbed_q)
        weights = q_res @ kv_cache.transpose(-2, -1)
        weights = weights * (self.head_size ** -0.5)
        weights = self._apply_masks(weights, mask, past_tokens)
        weights_normalized = F.softmax(weights, dim=-1)
        weights_normalized = self.dropout(weights_normalized)

        compressed_v = self.W_v(kv_cache)
        out = weights_normalized @ compressed_v
        return out, kv_cache


class MHLA(nn.Module):
    def __init__(
        self,
        embeddings_dims=ModelArgs.embeddings_dims,
        no_of_heads=ModelArgs.no_of_heads,
        latent_dim=ModelArgs.latent_dim,
        attn_dropout=ModelArgs.attn_dropout,
        device=ModelArgs.device,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                LatentAttention(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    latent_dim=latent_dim,
                    attn_dropout=attn_dropout,
                    device=device,
                )
                for _ in range(no_of_heads)
            ]
        )
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)

    def forward(self, x, kv_cache=None, mask=None):
        if kv_cache is None:
            kv_cache = [None for _ in range(len(self.heads))]

        res = []
        new_cache = []
        for i in range(len(self.heads)):
            head_out, head_cache = self.heads[i](x, kv_cache=kv_cache[i], mask=mask)
            res.append(head_out)
            new_cache.append(head_cache)

        concat=torch.cat(res, dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out, new_cache


class FFN(nn.Module):
    def __init__(self, embeddings_dims=ModelArgs.embeddings_dims, dropout=ModelArgs.dropout, device=ModelArgs.device):
        super().__init__()
        self.linear_layer = nn.Linear(embeddings_dims, embeddings_dims, device=device)
        self.linear_layer2 = nn.Linear(embeddings_dims, embeddings_dims, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_layer(x)
        x = F.gelu(x)
        x = self.linear_layer2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embeddings_dims=ModelArgs.embeddings_dims,
        no_of_heads=ModelArgs.no_of_heads,
        latent_dim=ModelArgs.latent_dim,
        experts=ModelArgs.experts,
        top_experts=ModelArgs.top_experts,
        dropout=ModelArgs.dropout,
        attn_dropout=ModelArgs.attn_dropout,
        noisy_topk=ModelArgs.noisy_topk,
        use_shared_expert=ModelArgs.use_shared_expert,
        useauxFreeLoadBalancingLoss=ModelArgs.useauxFreeLoadBalancingLoss,
        aux_free_bias_update_rate=ModelArgs.aux_free_bias_update_rate,
        device=ModelArgs.device,
    ):
        super().__init__()
        self.mha = MHLA(
            embeddings_dims=embeddings_dims,
            no_of_heads=no_of_heads,
            latent_dim=latent_dim,
            attn_dropout=attn_dropout,
            device=device,
        )
        self.layer_norm1 =Normalization(embeddings_dims=embeddings_dims, device=device)
        self.layer_norm2 =Normalization(embeddings_dims=embeddings_dims, device=device)
        self.dropout=nn.Dropout(p=dropout)
        self.moe_block = MoeLayer(
            embeddings_size=embeddings_dims,
            experts=experts,
            top_experts=top_experts,
            noisy_topk=noisy_topk,
            use_shared_expert=use_shared_expert,
            useauxFreeLoadBalancingLoss=useauxFreeLoadBalancingLoss,
            aux_free_bias_update_rate=aux_free_bias_update_rate,
            device=device,
        )

    def forward(self, x, kv_cache=None, mask=None):
        out, kv_cache = self.mha(self.layer_norm1(x), kv_cache=kv_cache, mask=mask)
        x = x + out
        x = x + self.moe_block(self.layer_norm2(x))
        return x, kv_cache


class Block(nn.Module):
    def __init__(
        self,
        vocab_size=ModelArgs.vocab_size,
        embeddings_dims=ModelArgs.embeddings_dims,
        no_of_decoder_layers=ModelArgs.no_of_decoder_layers,
        no_of_heads=ModelArgs.no_of_heads,
        latent_dim=ModelArgs.latent_dim,
        experts=ModelArgs.experts,
        top_experts=ModelArgs.top_experts,
        dropout=ModelArgs.dropout,
        attn_dropout=ModelArgs.attn_dropout,
        noisy_topk=ModelArgs.noisy_topk,
        use_shared_expert=ModelArgs.use_shared_expert,
        useauxFreeLoadBalancingLoss=ModelArgs.useauxFreeLoadBalancingLoss,
        aux_free_bias_update_rate=ModelArgs.aux_free_bias_update_rate,
        device=ModelArgs.device,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings_dims, device=device)
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    latent_dim=latent_dim,
                    experts=experts,
                    top_experts=top_experts,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    noisy_topk=noisy_topk,
                    use_shared_expert=use_shared_expert,
                    useauxFreeLoadBalancingLoss=useauxFreeLoadBalancingLoss,
                    aux_free_bias_update_rate=aux_free_bias_update_rate,
                    device=device,
                )
                for _ in range(no_of_decoder_layers)
            ]
        )
        self.linear_layer = nn.Linear(embeddings_dims, vocab_size, bias=False, device=device)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = Normalization(embeddings_dims=embeddings_dims, device=device)

        self.apply(self._init_weights)
        self.linear_layer.weight = self.embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, kv_cache=None, mask=None, return_cache=False):
        if kv_cache is None:
            kv_cache = [None for _ in range(len(self.decoder))]

        new_cache = []
        for i in range(len(self.decoder)):
            x, layer_cache = self.decoder[i](x, kv_cache=kv_cache[i], mask=mask)
            new_cache.append(layer_cache)

        x = self.dropout(x)
        x = 2 * (len(self.decoder) ** -0.5) * x
        x = self.norm(x)

        if return_cache:
            return x, new_cache
        return x


class DeepSeekV3(nn.Module):
    def __init__(
        self,
        vocab_size=ModelArgs.vocab_size,
        block_size=ModelArgs.block_size,
        embeddings_dims=ModelArgs.embeddings_dims,
        no_of_decoder_layers=ModelArgs.no_of_decoder_layers,
        no_of_heads=ModelArgs.no_of_heads,
        latent_dim=ModelArgs.latent_dim,
        experts=ModelArgs.experts,
        top_experts=ModelArgs.top_experts,
        mtp_heads=ModelArgs.mtp_heads,
        dropout=ModelArgs.dropout,
        attn_dropout=ModelArgs.attn_dropout,
        noisy_topk=ModelArgs.noisy_topk,
        use_shared_expert=ModelArgs.use_shared_expert,
        useauxFreeLoadBalancingLoss=ModelArgs.useauxFreeLoadBalancingLoss,
        aux_free_bias_update_rate=ModelArgs.aux_free_bias_update_rate,
        device=ModelArgs.device,
    ):
        super().__init__()
        self.block_size = block_size
        self.mtp_heads = mtp_heads
        self.embeddings_dims = embeddings_dims

        self.decoder = Block(
            vocab_size=vocab_size,
            embeddings_dims=embeddings_dims,
            no_of_decoder_layers=no_of_decoder_layers,
            no_of_heads=no_of_heads,
            latent_dim=latent_dim,
            experts=experts,
            top_experts=top_experts,
            dropout=dropout,
            attn_dropout=attn_dropout,
            noisy_topk=noisy_topk,
            use_shared_expert=use_shared_expert,
            useauxFreeLoadBalancingLoss=useauxFreeLoadBalancingLoss,
            aux_free_bias_update_rate=aux_free_bias_update_rate,
            device=device,
        )
        self.embedding = self.decoder.embeddings
        self.pos_embeddings = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims,
            block_size=block_size,
            device=device,
        )

        self.norm1 = nn.LayerNorm(embeddings_dims, eps=1e-6, device=device)
        self.norm2 = nn.LayerNorm(embeddings_dims, eps=1e-6, device=device)
        self.linear_layer = nn.Linear(2 * embeddings_dims, embeddings_dims, device=device)
        self.unilayer = nn.ModuleList(
            [
                DecoderLayer(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    latent_dim=latent_dim,
                    experts=experts,
                    top_experts=top_experts,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    noisy_topk=noisy_topk,
                    use_shared_expert=use_shared_expert,
                    useauxFreeLoadBalancingLoss=useauxFreeLoadBalancingLoss,
                    aux_free_bias_update_rate=aux_free_bias_update_rate,
                    device=device,
                )
                for _ in range(mtp_heads)
            ]
        )

    def _cache_start_pos(self, kv_cache):
        if kv_cache is None:
            return 0
        if len(kv_cache) == 0 or kv_cache[0] is None:
            return 0
        if len(kv_cache[0]) == 0 or kv_cache[0][0] is None:
            return 0
        return kv_cache[0][0].shape[1]

    def forward(self, x, inference=False, mask=None, kv_cache=None, return_cache=False, start_pos=None):
        # x is [B,T] token ids
        if x.ndim != 2:
            raise ValueError("x has to be 2D tensor with token ids")

        if x.shape[1] > self.block_size:
            raise ValueError("sequence length is bigger than block_size")

        if start_pos is None:
            start_pos = self._cache_start_pos(kv_cache)

        x_embed= self.embedding(x)
        x_embed= x_embed + self.pos_embeddings(x_embed, start_pos=start_pos)

        if mask is not None and mask.ndim == 2:
            x_embed= x_embed * mask[:, : x_embed.shape[1]].unsqueeze(-1)

        decoder_out, new_cache = self.decoder(
            x_embed,
            kv_cache=kv_cache,
            mask=mask,
            return_cache=True,
        )

        if inference or self.mtp_heads == 0:
            logits = self.decoder.linear_layer(decoder_out)
            if return_cache:
                return logits, new_cache
            return logits

        B, T, C = decoder_out.shape
        if T <= self.mtp_heads:
            raise ValueError("sequence length has to be bigger than mtp_heads")

        valid_tokens = T - self.mtp_heads
        outputs = []
        context = decoder_out[:, :valid_tokens, :]

        for k in range(self.mtp_heads):
            embed = x_embed[:, k + 1 : k + 1 + valid_tokens, :]
            h_z = self.norm2(context)
            embed = self.norm1(embed)
            combined = torch.cat([embed, h_z], dim=-1)
            merged = self.linear_layer(combined)

            mtp_mask = None
            if mask is not None and mask.ndim == 2:
                mtp_mask = mask[:, :valid_tokens]

            merged, _ = self.unilayer[k](merged, mask=mtp_mask)
            logits = self.decoder.linear_layer(merged)
            outputs.append(logits)

        final_output = torch.stack(outputs, dim=2)
        if return_cache:
            return final_output, new_cache
        return final_output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randint(0, 1000, (2, 8), device=device)

    model= DeepSeekV3(
        vocab_size=1000,
        block_size=8,
        embeddings_dims=32,
        no_of_decoder_layers=2,
        no_of_heads=4,
        latent_dim=8,
        experts=4,
        top_experts=2,
        mtp_heads=2,
        device=device,
    )

    train_logits = model(x)
    infer_logits = model(x, inference=True)
    print("one forward pass training shape", train_logits.shape)
    print("one forward pass inference shape", infer_logits.shape)
