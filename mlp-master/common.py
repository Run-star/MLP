
def box_muller_transform(u1, u2):

    z1 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
    z2 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
    return z1, z2

class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 from Uniform(0, 1), i.e. interval [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def rand(self, n, a=0, b=1):
        # return n random float32 from Uniform(a, b), in a list
        return [self.random() * (b - a) + a for _ in range(n)]

    def randn(self, n, mu=0, sigma=1):
        # return n random float32 from Normal(0, 1), in a list
        # (note box-muller transform returns two numbers at a time)
        out = []
        for _ in range((n + 1) // 2):
            u1, u2 = self.random(), self.random()
            z1, z2 = box_muller_transform(u1, u2)
            out.extend([z1 * sigma + mu, z2 * sigma + mu])
        out = out[:n] # if n is odd crop list
        return out

class StepTimer:
    def __init__(self, ema_alpha=0.9):
        self.ema_alpha = ema_alpha
        self.ema_time = 0
        self.corrected_ema_time = 0.0
        self.start_time = None
        self.step = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        iteration_time = end_time - self.start_time
        self.ema_time = self.ema_alpha * self.ema_time + (1 - self.ema_alpha) * iteration_time
        self.step += 1
        self.corrected_ema_time = self.ema_time / (1 - self.ema_alpha ** self.step) # bias correction

    def get_dt(self):
        return self.corrected_ema_time
