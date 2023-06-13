import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder

class LEDInitializer(nn.Module):
	def __init__(self, t_obs=8, s=2, n=20):
		super(LEDInitializer, self).__init__()
		self.s, self.n = s, n
		self.input_dim = t_obs * 6
		self.hidden_dim = self.input_dim * 4
		self.output_dim = s * n
		self.fut_len = s // 2

		self.social_encoder = social_transformer(t_obs)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, s, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, mask=None):
		'''
		x: batch size, t_p, 6
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256
		
		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		# B, 256

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale



