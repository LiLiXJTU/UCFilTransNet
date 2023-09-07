class CTrans_Attention(nn.Module):
    def __init__(self, dim,h,w,heads=4,  attn_drop=0.):
        super(CTrans_Attention, self).__init__()
        self.num_attention_heads = heads

        self.dim_head = int(dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.dim_head
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)

        self.sigmoid = nn.Sigmoid()
    def forward(self, q, x):
        B, N, C = x.shape  
        BH, NH, CH = q.shape  
        aH = bH = int(math.sqrt(NH))
        q = q.view(BH, aH, bH, CH)
        q = q.to(torch.float32)
        q = torch.fft.rfft2(q, dim=(1, 2), norm='ortho')
        weight_q = torch.view_as_complex(self.complex_weight)
        q = q * weight_q
        q = torch.fft.irfft2(q, s=(aH, bH), dim=(1, 2), norm='ortho')

        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C)
        x = x.permute(0,3,1,2)
        x = F.interpolate(x,size=(aH,bH))
        x = x.permute(0,2,3,1)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight_x = torch.view_as_complex(self.complex_weight)
        x = x * weight_x
        x = torch.fft.irfft2(x, s=(aH, bH), dim=(1, 2), norm='ortho')

        mixed_query_layer = q
        mixed_key_layer = x
        mixed_value_layer = x
        mixed_query_layer = rearrange(mixed_query_layer, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads,
                      h=aH, w=bH)
        mixed_key_layer, mixed_value_layer = map(lambda t: rearrange(t, 'b h w (dim_head heads) -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads, h=aH, w=bH), (mixed_key_layer, mixed_value_layer))

        attention_scores = torch.einsum('bhid,bhjd->bhij', mixed_query_layer, mixed_key_layer)
        attention_scores = attention_scores / math.sqrt(self.dim_head)
        attention_probs = self.sigmoid(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.einsum('bhij,bhjd->bhid', attention_probs, mixed_value_layer)

        context_layer = rearrange(context_layer, 'b heads (h w) dim_head -> b h w (dim_head heads)', dim_head=self.dim_head, heads=self.num_attention_heads,
                      h=aH, w=bH)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = attention_output.reshape(BH, NH, CH)
        return attention_output