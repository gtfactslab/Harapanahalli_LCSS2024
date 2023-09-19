import numpy as np
from math import ceil

class TimeSpec :
    def __init__(self, type, t_step, u_step) -> None:
        # Discrete or Continuous
        self.type = type
        # Euler Integration step
        self.t_step = t_step
        # Control Update time
        self.u_step = u_step

    def lentt (self, ti, tf) :
        return ceil((tf + self.t_step - ti)/self.t_step)

    def tt (self, ti, tf) :
        return np.arange(ti, tf + self.t_step, self.t_step)
    
    def tu (self, ti, tf) :
        return np.arange(ti, tf, self.t_step).reshape((-1,round(self.u_step/self.t_step)))
    
    def lenuu (self, ti, tf) :
        return ceil((tf + self.u_step - ti)/self.u_step)

    def uu (self, ti, tf) :
        return np.arange(ti, tf + self.u_step, self.u_step)
    
class DiscreteTimeSpec (TimeSpec) :
    def __init__(self) -> None:
        super().__init__('discrete', 1, 1)
    def __str__(self) -> str:
        return f'Discrete'

class DiscretizedTimeSpec (TimeSpec) :
    def __init__(self, t_step) -> None:
        super().__init__('discretized', t_step, t_step)

    def __str__(self) -> str:
        return f'Discretized (t_step: {self.t_step})'

class ContinuousTimeSpec (TimeSpec) :
    def __init__(self, t_step, u_step) -> None:
        if t_step > u_step :
            raise Exception('t_step should be smaller than u_step in ContinuousTimeSpec')
        super().__init__('continuous', t_step, u_step)

    def __str__(self) -> str:
        return f'Continuous (t_step: {self.t_step}, u_step: {self.u_step})'
