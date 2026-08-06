import torch as th
from torchdiffeq import odeint
from .utils import time_shift, get_lin_function

class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        do_shift=False,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.do_shift = do_shift
        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        x = x.float()
        device = x[0].device if isinstance(x, tuple) else x.device

        def _infer_shift_seq_len(x_tensor, kwargs):
            """
            Infer token/sequence length for time-shift scheduling.
            For image latents we typically use [B, N, C], while motion code uses
            [L, B, C]. Avoid coupling shift to batch size by checking a batch hint
            from model kwargs.
            """
            if x_tensor.dim() < 2:
                return int(x_tensor.shape[0])

            batch_hint = None
            for key in ("text_tokens", "image_latents", "modality_positions", "motion_modality_positions"):
                val = kwargs.get(key, None)
                if isinstance(val, th.Tensor) and val.dim() >= 1:
                    batch_hint = int(val.shape[0])
                    break

            if batch_hint is not None:
                if int(x_tensor.shape[0]) == batch_hint and x_tensor.dim() >= 2:
                    return int(x_tensor.shape[1])  # [B, N, ...]
                if int(x_tensor.shape[1]) == batch_hint:
                    return int(x_tensor.shape[0])  # [L, B, ...]

            # Fallback keeps previous behavior for unknown layouts.
            return int(x_tensor.shape[1])

        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs).float()
            return model_output
        
        t = self.t.to(device)
        if self.do_shift:
            shift_input = x[0] if isinstance(x, tuple) else x
            shift_seq_len = _infer_shift_seq_len(shift_input, model_kwargs)
            mu = get_lin_function(y1=0.5, y2=1.15)(shift_seq_len)
            t = time_shift(mu, 1.0, t)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
