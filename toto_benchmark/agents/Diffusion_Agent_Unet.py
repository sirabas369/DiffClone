import os
import pickle
import torch
from torch import nn
from .BaseAgent import BaseAgent
# from .denoising_transformer import transformer_encoder
from .unet1d import ConditionalUnet1D
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from .Agent import Agent
from tqdm import tqdm


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def get_stats(arr):
    arr_std = arr.std(0)
    arr_std[arr_std < 1e-4] = 1
    return len(arr_std), arr.mean(0), arr_std


class DMAgent(Agent):
    def __init__(self, models, learning_rate, device, out_dim, len_dataloader, num_epochs, H=1):

        self.H, self.t, self.cache, self.out_dim = H, 0, None, out_dim
        self.models = models
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.loss_reduction = torch.mean
        self.parameters = sum([list(m.parameters()) for m in self.models.values()], [])
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate, weight_decay=1e-6)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type='epsilon') 
        # self.ddim_scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type='epsilon') 
        if type(len_dataloader) is int:
            self.lr_scheduler = get_scheduler(name='cosine', optimizer=self.optimizer, num_warmup_steps=500, num_training_steps=len_dataloader * num_epochs)
        self.device = device
        self.epoch = 0
        self.loaded_epoch = 0
        self.item_losses = None
        self.loss_dict = {}

    def save(self, foldername, filename='Agent.pth'):
        state = {'epoch': self.epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        for mname, m in self.models.items():
            state[mname] = m.state_dict()
            m.save_stats(foldername)
        torch.save(state, os.path.join(foldername, str(self.epoch)+filename))

    def load(self, foldername, device=None, filename='Agent.pth'):
        if device is not None:
            self.device = device
        checkpoint = torch.load(os.path.join(foldername, filename), map_location=torch.device(self.device))
        self.epoch = checkpoint['epoch']
        self.loaded_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for mname, m in self.models.items():
            if mname in checkpoint:
                m.load_state_dict(checkpoint[mname])
                m = m.to(self.device)
            else:
                m = m.to(self.device)
                print(f"Not loading {mname} from checkpoint")

        self.models.to(self.device)

    def zero_grad(self):
        [m.zero_grad() for m in self.models.values()] 
        self.optimizer.zero_grad()

    def train(self, sample, epoch):
        self.epoch = epoch
        [m.train() for m in self.models.values()]
        self.zero_grad()
        self.compute_loss(sample)
        self.loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return self.loss.item()

    def eval(self, sample):
        [m.eval() for m in self.models.values()]
        with torch.no_grad():
            self.compute_loss(sample)
            return self.loss.item()

    def pack_one_batch(self, sample):
        for k, v in sample.items():
            t = v if torch.is_tensor(v) else torch.from_numpy(v)
            if t.dim() == 1:
                sample[k] = t.float().unsqueeze(0).to(self.device)
            else:
                sample[k] = t.float().to(self.device)

        return sample


    def forward(self, sample):
        return self.models['decoder'](sample, self.noise_scheduler, self.H, self.epoch)

    def diversity_loss(self, outputs, epsilon=1e-8):
        """
        Encourage diversity in the batch of sequences by penalizing similarity.
        :param outputs: Tensor of shape [Batch, Sequence Length, Action Dimension]
        :param epsilon: Small value to avoid division by zero
        :return: Diversity loss value
        """
        batch_size = outputs.size(0)
        seq_length = outputs.size(1)
        action_dim = outputs.size(2)
        
        # Flatten the sequence and action dimensions
        outputs_flat = outputs.view(batch_size, -1)  # [Batch, Sequence Length * Action Dimension]
        
        # Normalize the flat outputs to ensure cosine similarity is between 0 and 1
        outputs_flat = outputs_flat / (torch.norm(outputs_flat, dim=1, keepdim=True) + epsilon)
        
        # Compute the pairwise cosine similarity between sequences in the batch
        similarity = torch.matmul(outputs_flat, outputs_flat.transpose(0, 1))  # [Batch, Batch]
        
        # Ensure the similarity values are between 0 and 1
        similarity = (similarity + 1.0) / 2.0
        
        # Create a mask to exclude the diagonal elements
        mask = torch.eye(batch_size, device=outputs.device).bool()
        
        # Apply the mask to exclude diagonal elements from the sum
        non_diag_mask = ~mask
        similarity = similarity.masked_select(non_diag_mask)
        
        # Calculate the diversity loss as the mean of the non-diagonal elements of the similarity matrix
        div_loss = 1/torch.mean(similarity)
        
        return div_loss

    def compute_loss(self, sample):

        pred, labels = self.forward(sample)
        abs_loss = self.loss_reduction(self.loss_fn(pred, labels))
    
        # Try adding Diversity Loss
        # div_loss = self.diversity_loss( pred.reshape(pred.shape[0], self.H, -1) )
        # losses = abs_loss + 0.01*div_loss
        losses = abs_loss
        self.loss = losses
        self.loss_dict['abs_loss'] = abs_loss
        self.loss_dict['div_loss'] = 0


    def sampling(self, inp):
        return self.models['decoder'].sampling(inp, self.noise_scheduler, self.H)

    def predict(self, sample):
        if self.H == 1:
            [m.eval() for m in self.models.values()]
            with torch.no_grad():
                sample = self.pack_one_batch(sample)
                output = self.sampling(sample).to('cpu').detach().numpy()
                return output
        else:
            index = self.t % self.H//2
            if index == 0:
                [m.eval() for m in self.models.values()]
                with torch.no_grad():
                    sample = self.pack_one_batch(sample)
                    # output = self.sampling(sample).to('cpu').detach().numpy()
                    output = self.sampling(sample)
                    self.cache = output.reshape([self.H, -1])

            self.t += 1
            return self.cache[index].to('cpu').detach().numpy()


class Policy(nn.Module):
    def __init__(self, inp_dim, out_dim, H, hidden_dim=128, device="cuda"):
        super(Policy, self).__init__()

        self.inp_dim, self.out_dim, self.H, self.hidden_dim = inp_dim, out_dim, H, hidden_dim
        self.model = ConditionalUnet1D(input_dim=out_dim, global_cond_dim=inp_dim).to(device)
        self.device = device
        self.register_buffer("inp_mean", torch.zeros(inp_dim))
        self.register_buffer("inp_std", torch.ones(inp_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim*H))
        self.register_buffer("out_std", torch.ones(out_dim*H))

    def set_stats(self, dataset):
        inp_dim, inp_mean, inp_std = get_stats(dataset.inputs)
        _, out_mean, out_std = get_stats(dataset.labels)

        self.inp_mean[:inp_dim].copy_(inp_mean)
        self.inp_std[:inp_dim].copy_(inp_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def save_stats(self, foldername, filename='policy_stats.pkl'):
        policy_stats = {
            'inp_mean': self.inp_mean,
            'inp_std': self.inp_std,
            'out_mean': self.out_mean,
            'out_std': self.out_std
        }
        with open(os.path.join(foldername, filename), 'wb') as handle:
            pickle.dump(policy_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_stats(self, foldername, filename='policy_stats.pkl'):
        policy_stats = pickle.load(open(os.path.join(foldername, filename), 'rb'))
        self.inp_mean.copy_(policy_stats['inp_mean'])
        self.inp_std.copy_(policy_stats['inp_std'])
        self.out_mean.copy_(policy_stats['out_mean'])
        self.out_std.copy_(policy_stats['out_std'])

    def sampling(self, inp, noise_scheduler, H):
        noise_scheduler.set_timesteps(50)
        n_samples = 1
        samples = torch.randn(n_samples, self.H, self.out_dim).to(self.device)
        observations = inp["inputs"]

        observations = (observations - self.inp_mean) / self.inp_std

        for i,t in tqdm(enumerate(noise_scheduler.timesteps)): 
            with torch.no_grad():
                timesteps = torch.ones(n_samples, ) * t
                timesteps = timesteps.to(self.device).long()
                residual = self.model(samples, timesteps.float(), observations) 

                samples = noise_scheduler.step(residual, t, samples).prev_sample 

        samples = samples.reshape(n_samples, -1)    
        actions = self.out_mean + self.out_std * samples
        return actions

    def forward(self, samples, noise_scheduler, H, epoch):
        observations = samples["inputs"]
        actions = samples["labels"]    
        observations = (observations - self.inp_mean) / self.inp_std
        actions = (actions - self.out_mean) / self.out_std
        h = actions.reshape(actions.shape[0], H, -1)
        ## Denoising training
        bs = h.shape[0] 
        noise = torch.randn(h.shape).to(self.device)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=self.device).long()
        noisy_h = noise_scheduler.add_noise(h, noise, timesteps)
        pred = self.model(noisy_h, timesteps, global_cond=observations)

        return pred, noise

def _init_agent_from_config(config, device='cuda', normalization=None):
    torch.manual_seed(0)
    if 'H' in config.data.keys():
        H = config.data.H
    else:
        H = 1

    if 'hidden_dim' in config['agent']:
        hidden_dim = config.agent.hidden_dim
    else:
        hidden_dim = 128
    models = nn.ModuleDict({
        'decoder': Policy(
            config.data.in_dim,
            config.data.out_dim,
            H,
            hidden_dim,
            device)})

    if normalization is not None:
        models['decoder'].set_stats(normalization)
    else:
        print(os.path.join(config.saved_folder, 'policy_stats.pkl'))
        assert os.path.isfile(os.path.join(config.saved_folder, 'policy_stats.pkl'))
        models['decoder'].load_stats(config.saved_folder)

    for k,m in models.items():
        m.to(device)
        if k=="img_encoder" and config.model.use_resnet:
            print("*** Resnet image encoder, do not init weight")
        else:
            m.apply(init_weights)

    bc_agent = DMAgent(models, config.training.lr, device, config.data.out_dim, config.training.len_dataloader, config.training.epochs, H)
    # load weights if exist (during inference)
    if os.path.isfile(os.path.join(config.saved_folder, 'Agent.pth')):
        bc_agent.load(config.saved_folder)

    return bc_agent, None # image transforms is None for BCAgent