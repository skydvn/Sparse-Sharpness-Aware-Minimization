import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class SAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
    
    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
        }
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        
        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


@OPTIMIZER_REGISTRY.register()
class FSSAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho, rho_schedule="constant",
                 rho_min=None, rho_max=None, rho_update_freq=1, total_epochs=None) -> None:
        """
        FSSAM optimizer with adaptive rho scheduling.

        Args:
            params: model parameters
            base_optimizer: underlying optimizer (e.g., SGD, Adam)
            rho: initial perturbation radius
            rho_schedule: schedule type - "constant", "cosine", "linear_decay", "linear_growth", "cyclic"
            rho_min: minimum rho value (default: rho * 0.5)
            rho_max: maximum rho value (default: rho * 2.0)
            rho_update_freq: frequency of rho updates in epochs
            total_epochs: total training epochs (needed for some schedules)
        """
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        self.initial_rho = rho
        self.rho_schedule = rho_schedule
        self.rho_min = rho_min if rho_min is not None else rho * 0.5
        self.rho_max = rho_max if rho_max is not None else rho * 2.0
        self.rho_update_freq = rho_update_freq
        self.total_epochs = total_epochs
        self.current_epoch = 0

        super(FSSAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["rho_schedule"] = rho_schedule
            group["rho_update_freq"] = rho_update_freq

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho,
            "rho_schedule": getattr(args, "rho_schedule", "constant"),
            "rho_min": getattr(args, "rho_min", None),
            "rho_max": getattr(args, "rho_max", None),
            "rho_update_freq": getattr(args, "rho_update_freq", 1),
            "total_epochs": getattr(args, "epochs", None),
        }

    @torch.no_grad()
    def update_rho(self, epoch):
        """
        Update rho based on the schedule and current epoch.

        Args:
            epoch: current training epoch
        """
        self.current_epoch = epoch

        if epoch % self.rho_update_freq != 0:
            return self.rho

        if self.rho_schedule == "constant":
            new_rho = self.initial_rho

        elif self.rho_schedule == "cosine":
            # Cosine annealing from rho_max to rho_min
            if self.total_epochs is None:
                raise ValueError("total_epochs must be specified for cosine schedule")
            progress = min(epoch / self.total_epochs, 1.0)
            new_rho = self.rho_min + (self.rho_max - self.rho_min) * 0.5 * (1 + math.cos(math.pi * progress))

        elif self.rho_schedule == "linear_decay":
            # Linear decay from initial_rho to rho_min
            if self.total_epochs is None:
                raise ValueError("total_epochs must be specified for linear_decay schedule")
            progress = min(epoch / self.total_epochs, 1.0)
            new_rho = self.initial_rho - (self.initial_rho - self.rho_min) * progress

        elif self.rho_schedule == "linear_growth":
            # Linear growth from initial_rho to rho_max
            if self.total_epochs is None:
                raise ValueError("total_epochs must be specified for linear_growth schedule")
            progress = min(epoch / self.total_epochs, 1.0)
            new_rho = self.initial_rho + (self.rho_max - self.initial_rho) * progress

        elif self.rho_schedule == "cyclic":
            # Cyclic between rho_min and rho_max with period of rho_update_freq * 10 epochs
            cycle_length = self.rho_update_freq * 10
            progress = (epoch % cycle_length) / cycle_length
            new_rho = self.rho_min + (self.rho_max - self.rho_min) * 0.5 * (1 + math.cos(2 * math.pi * progress))

        elif self.rho_schedule == "step_decay":
            # Step decay: reduce by half every rho_update_freq epochs
            decay_steps = epoch // self.rho_update_freq
            new_rho = max(self.initial_rho * (0.5 ** decay_steps), self.rho_min)

        elif self.rho_schedule == "warmup_cosine":
            # Warmup for 10% then cosine decay
            if self.total_epochs is None:
                raise ValueError("total_epochs must be specified for warmup_cosine schedule")
            warmup_epochs = int(0.1 * self.total_epochs)
            if epoch < warmup_epochs:
                new_rho = self.rho_min + (self.initial_rho - self.rho_min) * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (self.total_epochs - warmup_epochs)
                new_rho = self.rho_min + (self.initial_rho - self.rho_min) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown rho_schedule: {self.rho_schedule}")

        # Update rho in optimizer
        self.rho = new_rho
        print(f"================= Rho update to {self.rho} =================")
        for group in self.param_groups:
            group["rho"] = new_rho

        return new_rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, epoch=None, batch_idx=None, logger=None, **kwargs):
        """
        Perform a single optimization step with adaptive rho.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            epoch: Current training epoch
            batch_idx: Current batch index
            logger: Logger object for logging rho updates
        """
        assert closure is not None, "SAM requires closure, which is not provided."

        # Update rho at the beginning of each epoch (batch_idx == 0)
        if epoch is not None and batch_idx is not None and batch_idx == 0:
            old_rho = self.rho
            new_rho = self.update_rho(epoch)
            if logger is not None and abs(new_rho - old_rho) > 1e-6:
                logger.log(f'Epoch {epoch}: Updated rho from {old_rho:.6f} to {new_rho:.6f}')

        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


@OPTIMIZER_REGISTRY.register()
class SSAMF(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, rho, sparsity, num_samples, update_freq) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 1.0 <= num_samples, f"num_samples should be greater than 1: {num_samples}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        super(SSAMF, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["num_samples"] = num_samples
            group["update_freq"] = update_freq

        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "num_samples": args.num_samples,
            "update_freq": args.update_freq,
        }
    
    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).to(p)

    @torch.no_grad()
    def update_mask(self, model, train_data, **kwargs):
        fisher_value_dict = {}
        fisher_mask_dict = {}
        for group in self.param_groups:
            for p in group['params']:
                fisher_value_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)
                fisher_mask_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)

        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=1,
            num_workers=4,
            shuffle=True,
        )
        # cal fisher value
        with torch.enable_grad():
            for idx, (image, label) in enumerate(train_dataloader):
                if idx >= self.num_samples: break
                if idx % (self.num_samples // 10) == 0: print('Updating Mask: [{}/{}]..'.format(idx, self.num_samples))
                image, label = image.cuda(), label.cuda()
                
                output = model(image)
                loss = criterion(output, label)
                loss.backward()

                for group in self.param_groups:
                    for p in group["params"]:
                        fisher_value_dict[id(p)] += torch.square(p.grad).data
                model.zero_grad()
        
        # topk fisher value 
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])
        
        keep_num = int(len(fisher_value_list) * (1 - self.sparsity))
        _value, _index = torch.topk(fisher_value_list, keep_num)
        
        mask_list = torch.zeros_like(fisher_value_list)
        mask_list.scatter_(0, _index, torch.ones_like(_value))

        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, model=None, epoch=None, batch_idx=None, train_data=None, logger=None, **kwargs):
        super().step(closure, **kwargs)
        assert model is not None
        assert train_data is not None
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(model, train_data)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
            
    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num

@OPTIMIZER_REGISTRY.register()
class SSAMD(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, 
        rho, sparsity, drop_rate, drop_strategy, growth_strategy, update_freq, T_start, T_end) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 0.0 <= drop_rate <= 1.0, f"drop_rate should between 0 and 1: {drop_rate}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy
        self.update_freq = update_freq
        self.T_start = T_start
        self.T_end = T_end
        super(SSAMD, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["drop_rate"] = drop_rate
            group["drop_strategy"] = drop_strategy
            group["growth_strategy"] = growth_strategy
            group["update_freq"] = update_freq
            group["T_end"] = T_end
        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "drop_rate": args.drop_rate,
            "drop_strategy": args.drop_strategy,
            "growth_strategy": args.growth_strategy,
            "update_freq": args.update_freq,
            "T_end": args.epochs,
            "T_start": 0,
        }
    
    @torch.no_grad()
    def init_mask(self):
        random_scores = []
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['score'] = torch.rand(size=p.shape).cpu().data
                random_scores.append(self.state[p]['score'])
        random_scores = torch.cat([torch.flatten(x) for x in random_scores])
        live_num = len(random_scores) - math.ceil(len(random_scores) *self.sparsity)
        _value, _index = torch.topk(random_scores, live_num)

        mask_list = torch.zeros_like(random_scores)
        mask_list.scatter_(0, _index, torch.ones_like(_value))
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                del self.state[p]['score']
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
        
    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch):
        dr = (self.drop_rate) * (1 + math.cos(math.pi * (float(epoch - self.T_start) / (self.T_end - self.T_start)))) / 2 
        return dr           

    @torch.no_grad()
    def update_mask(self, epoch, **kwargs):
        death_scores = []
        growth_scores =[]
        for group in self.param_groups:
            for p in group['params']:
                death_score = self.get_score(p, self.drop_strategy)
                death_scores.append((death_score + 1e-7) * self.state[p]['mask'].cpu().data)

                growth_score = self.get_score(p, self.growth_strategy)
                growth_scores.append((growth_score + 1e-7) * (1 - self.state[p]['mask'].cpu().data))
        '''
            Death 
        '''
        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity)* death_rate, len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores, int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))
        '''
            Growth
        '''
        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)
        
        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

        '''
            Mask
        '''
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                death_mask = death_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                growth_mask = growth_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                
                self.state[p]['mask'] = death_mask + growth_mask
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
                
                
                    
        assert start_index == len(death_mask_list)

    def get_score(self, p, score_model=None):
        if score_model == 'weight':
            return torch.abs(p.clone()).cpu().data
        elif score_model == 'gradient':
            return torch.abs(p.grad.clone()).cpu().data
        elif score_model == 'random':
            return torch.rand(size=p.shape).cpu().data
        else:
            raise KeyError    
  
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, epoch=None, batch_idx=None, logger=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None

        self.first_step()
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(epoch)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.second_step()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num


@OPTIMIZER_REGISTRY.register()
class FLSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, args, **kwargs):
        # Extract parameters from args
        rho = getattr(args, 'rho', 0.05)
        sigma = getattr(args, 'sigma', 1.0)
        lmbda = getattr(args, 'lmbda', 0.9)
        adaptive = getattr(args, 'adaptive', False)

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FLSAM, self).__init__(params, defaults)

        # base_optimizer is already instantiated - just use it directly
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        # Ensure rho and adaptive are in all param groups
        for group in self.param_groups:
            group.setdefault('rho', rho)
            group.setdefault('adaptive', adaptive)

        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        print('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.clone()
                if "momentum" not in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            adaptive = group.get("adaptive", False)

            for p in group["params"]:
                if p.grad is None: continue
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@OPTIMIZER_REGISTRY.register()
class ASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, args, **kwargs):
        # Extract parameters from args
        rho = getattr(args, 'rho', 0.05)
        adaptive = getattr(args, 'adaptive', False)

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ASAM, self).__init__(params, defaults)

        # FIX: base_optimizer is already instantiated - don't call it
        self.base_optimizer = base_optimizer  # ← Changed this line
        self.param_groups = self.base_optimizer.param_groups

        # Ensure rho and adaptive are in all param groups
        for group in self.param_groups:
            group.setdefault('rho', rho)
            group.setdefault('adaptive', adaptive)

        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            adaptive = group.get("adaptive", False)

            for p in group["params"]:
                if p.grad is None: continue
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@OPTIMIZER_REGISTRY.register()
class NSAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, args, **kwargs):
        rho = getattr(args, 'rho', 0.05)
        adaptive = getattr(args, 'adaptive', False)

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(NSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        # Ensure rho and adaptive are in all param groups
        for group in self.param_groups:
            group.setdefault('adaptive', adaptive)
            group.setdefault('rho', rho)

        self.defaults.update(self.base_optimizer.defaults)
        self.has_normal = False
        self.args = args

    def step(self, model=None, images=None, targets=None, indices=None,
             criterion=None, epoch=None, step=None, batch_idx=None,
             train_data=None, logger=None, **kwargs):
        """
        NSAM step with mixed normal/SAM optimization
        Returns: (loss, acc1, acc5) for logging
        """
        device = images.device

        # Get flat sample indices
        flat_indices = self._get_flat_indices(
            indices=indices,
            epoch=epoch,
            step=step,
            train_data=train_data
        )

        # Split into normal and flat samples
        flat_set = set(flat_indices.tolist() if torch.is_tensor(flat_indices) else flat_indices)
        indices_list = indices.tolist() if torch.is_tensor(indices) else list(indices)

        normal_mask = torch.tensor([idx not in flat_set for idx in indices_list], device=device)
        normal_idx = torch.where(normal_mask)[0]
        flat_idx = torch.where(~normal_mask)[0]

        has_normal = len(normal_idx) > 0
        has_flat = len(flat_idx) > 0

        total_loss = 0.0

        # Process normal samples (standard SGD)
        if has_normal:
            self.zero_grad()
            normal_images = images[normal_idx]
            normal_targets = targets[normal_idx]

            output = model(normal_images)
            loss = criterion(output, normal_targets)
            total_loss += loss.item() * len(normal_idx)

            loss.backward()
            self.normal_step(zero_grad=False)

        # Process flat samples (SAM)
        if has_flat:
            if not has_normal:
                self.zero_grad()

            flat_images = images[flat_idx]
            flat_targets = targets[flat_idx]

            # First forward-backward
            output = model(flat_images)
            loss = criterion(output, flat_targets)
            total_loss += loss.item() * len(flat_idx)

            loss.backward()
            self.first_step(zero_grad=True)

            # Second forward-backward at perturbed point
            output = model(flat_images)
            loss = criterion(output, flat_targets)
            loss.backward()
            self.second_step(zero_grad=True)

        elif has_normal:
            # Only normal samples
            self.third_step(zero_grad=True)

        # Compute final loss and accuracy
        avg_loss = total_loss / len(images)

        with torch.no_grad():
            output = model(images)
            from utils.engine import accuracy
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        return torch.tensor(avg_loss), acc1, acc5

    def _get_flat_indices(self, indices, epoch, step, train_data):
        """
        Determine which samples should use SAM optimization
        """
        flat_ratio = getattr(self.args, 'flat_sample_ratio', 0.5)
        flat_selection = getattr(self.args, 'flat_selection', 'random')

        if flat_ratio <= 0:
            return torch.tensor([], dtype=torch.long)
        elif flat_ratio >= 1.0:
            return indices

        n_flat = int(len(indices) * flat_ratio)

        if flat_selection == 'random':
            perm = torch.randperm(len(indices))
            flat_idx = perm[:n_flat]
            return indices[flat_idx]

        elif flat_selection == 'curriculum':
            # Decrease SAM ratio over time
            max_epochs = getattr(self.args, 'epochs', 100)
            curriculum_ratio = flat_ratio * (1.0 - epoch / max_epochs)
            n_flat = int(len(indices) * max(curriculum_ratio, 0.1))
            perm = torch.randperm(len(indices))
            flat_idx = perm[:n_flat]
            return indices[flat_idx]

        else:
            perm = torch.randperm(len(indices))
            flat_idx = perm[:n_flat]
            return indices[flat_idx]

    @torch.no_grad()
    def normal_step(self, zero_grad=False):
        self.has_normal = True
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["normal_g"] = p.grad.clone()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            adaptive = group.get("adaptive", False)

            for p in group["params"]:
                if p.grad is None: continue
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                if self.has_normal:
                    # Combine gradients from normal and flat samples
                    p.grad = p.grad + self.state[p]["normal_g"]

        if self.has_normal:
            self.has_normal = False

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if self.has_normal and p in self.state and "normal_g" in self.state[p]:
                    p.grad = self.state[p]["normal_g"]

        if self.has_normal:
            self.has_normal = False

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups