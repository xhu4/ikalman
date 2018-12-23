from IPython.display import display
from bqplot import pyplot as plt
import time

import numpy as np
from scipy.linalg import expm
from ipywidgets import (Play, FloatSlider, Button, HBox, fixed, VBox,
                        interactive, IntSlider, jslink)


def queue(lst, num, max_len):
    """
    Add num to the end of lst. Keep the length not greater than max_len by
    removing the front part of the lst.
    """

    if lst.size == max_len:
        lst = np.roll(lst, -1)
        lst[-1] = num
    else:
        lst = np.append(lst, num)
        if lst.size > max_len:
            lst = lst[-max_len:]
    return lst


class Figure:
    """Controls a bqplot figure instance."""

    def __init__(self, animation_duration=100, aspect_ratio=1,
                 tail_len=1000, size=2):
        axes_options = {'x': {'label': 'x'}, 'y': {'label': 'y'}}
        self.fig = plt.figure(animation_duration=animation_duration,
                              min_aspect_ratio=aspect_ratio,
                              max_aspect_ratio=aspect_ratio)
        self.line = plt.plot([], [], marker_str='b-',
                             axes_options=axes_options)
        self.hline = plt.hline(0, opacities=[0], colors=[
                               'red'], stroke_width=3)
        self.scat = plt.plot([], [], 'ro')
        self.tail_len = tail_len
        plt.xlim(-size, size)
        plt.ylim(-size, size)

    def add_line(self, *args, **kwargs):
        return plt.plot(*args, figure=self.fig, **kwargs)

    def add_scat(self, *args, **kwargs):
        return plt.scatter(*args, figure=self.fig, **kwargs)

    def show(self):
        display(self.fig)

    def show_data(self, opac=0.5):
        self.hline.opacities = [opac]

    def update(self, val, tail=False, data=np.nan,
               estimate=None):
        self.scat.x = np.asarray(val[0]).reshape((-1,))
        self.scat.y = np.asarray([val[1]]).reshape((-1,))
        self.hline.y = np.array([data, data])
        if tail:
            self.line.x = queue(self.line.x, val[0], self.tail_len)
            self.line.y = queue(self.line.y, val[1], self.tail_len)
        else:
            self.clear_tail()

    def clear_tail(self):
        self.line.x = []
        self.line.y = []

    def clear(self):
        self.fig.clear()


class DynamicSystem:
    """A dynamic system.

    attributes:
    - dynamic:      a function that generates the next state based on current
                    state;
    - init_state:   initial state of the system;
    - state:        current state of the system.
    """

    def __init__(self, dynamic, v0):
        self.dynamic = dynamic
        self.init_state = np.array(v0).reshape((-1, 1))
        self.state = self.init_state

    def next(self, **kwargs):
        """Move to next state."""
        self.state = self.dynamic(self.state, **kwargs)
        return self.state

    def reset(self):
        self.state = self.init_state
        return self.state


class DynamicPlotter:
    """Plotting dynamic system."""

    def __init__(self, dynamic, v0,
                 observe=None, fig=None, **widgets):
        self.fig = Figure() if fig is None else fig

        self.dyn = DynamicSystem(dynamic, v0)
        self.observe = observe

        self.rst_bt = Button(description='Reset')
        self.obsv_bt = Button(description='Observe')
        self.rst_bt.on_click(self.reset)
        self.obsv_bt.on_click(self.show_data)
        self.buttons = HBox([self.rst_bt, self.obsv_bt])

        self.widgets = widgets

    def add_button(self, button):
        self.buttons.children += (button,)

    def show_data(self, button):
        self.fig.show_data()

    def set_obsv(self, observe):
        self.observe = observe

    def add_widgets(self, **widgets):
        self.widgets = {**self.widgets, **widgets}

    def rm_widgets(self, *keys):
        for key in keys:
            w = self.widgets.pop(key, None)
            if w:
                w.close()

    def reset(self, button):
        self.fig.clear_tail()
        self.fig.update(self.dyn.reset(), tail=True)

    def run(self, tail=True, sleep=0, **kwargs):
        m = self.dyn.next(**kwargs)
        data = self.observe(m, **kwargs)
        self.fig.update(m, tail=tail, data=data)
        time.sleep(sleep)

    def interact(self, **kwargs):
        w = interactive(self.run, tail=True, **self.widgets, **kwargs)
        display(HBox((self.fig.fig, VBox((self.buttons, w)))))


class KalmanFilter:
    def __init__(self, F, G, C, D, x0=None, S0=None, **useless):
        self.A = np.asarray(F)
        self.H = np.asarray(G)
        self.eye = np.eye(self.A.shape[0])
        self.sigma = np.asarray(C)
        self.Sigma = np.dot(self.sigma, self.sigma.T)
        self.gamma = np.asarray(D)
        self.m = np.asarray(x0)

    def build(self, dt):
        self.L = expm(self.A*dt)
        self.dt = dt

    def initialize(self, x0, S0=None):
        self.m = np.asarray(x0).reshape((-1, 1))
        self.C = np.asarray(S0)

    def filter(self, data):
        mhat = self.L@self.m
        chat = self.L@self.C@self.L.T + self.Sigma
        d = data - self.H@mhat
        K = self.dt*chat@self.H.T / \
            (self.H@chat@self.H.T*self.dt + np.dot(self.gamma, self.gamma.T))
        self.m = mhat + K@d
        self.C = (self.eye - K@self.H)@chat
        return self.m, self.C


def discrete(v, sigma, **useless):
    """A simple square discrete dynamic."""
    ret = np.array([[0, 1.0], [-1, 0]]) @ v
    ret += sigma*np.random.randn(2, 1)
    return ret


def continuous(v, sigma, dt, i, **useless):
    """A simple circular continuous dynamic."""
    theta = 0.865
    b = 1 + (theta*dt)**2
    b = 1/b
    A = np.array([[b, theta*dt*b], [-theta*dt*b, b]])
    B = (1-theta)*dt*np.array([[0, 2*np.pi], [-2*np.pi, 0]]) + np.eye(2)
    ret = A @ (B @ v.reshape((-1, 1)))
    ret += sigma*np.random.randn(2, 1)
    return ret


def _observe(state, dt=1, gamma=0, **useless):
    ret = state.copy()[1]
    ret += gamma * np.random.randn(1) / np.sqrt(dt)
    return ret


def _sigma_sld(value=0.0, min=0.0, max=0.4, step=0.01,
               description=r'\(\sigma\)'):
    """Create a slider for sigma."""
    return FloatSlider(value=value, min=min, max=max, step=step,
                       description=description)


def _spd_sld(value=0, min=0.01, max=0.3, step=0.01, description=r'slowness'):
    """Create a slider for dt (controls the speed)."""
    return FloatSlider(value=value, min=min, max=max, step=step,
                       description=description)


dsct = DynamicPlotter(discrete, [1, 1], observe=_observe, i=Play(),
                      sigma=_sigma_sld(),
                      gamma=_sigma_sld(description=r'\(\gamma\)'),
                      sleep=_spd_sld(value=0.5, max=1))


cnts = DynamicPlotter(continuous, [0, 1], observe=_observe,
                      fig=Figure(size=3),  i=Play(), sigma=_sigma_sld(),
                      gamma=_sigma_sld(description=r'\(\gamma\)'),
                      dt=fixed(0.05), sleep=_spd_sld())


def do_filter(init_guess=[0, 1], S0=np.eye(2), ic=[0, 1], N=1000,
              sigma=0.2, gamma=0.2, dt=0.1):
    KF = KalmanFilter(F=[[0, 1], [-1, 0]], G=[[0, 1]], C=sigma, D=gamma)
    KF.build(dt)

    x = np.empty((2, N))
    x[:, 0] = ic
    data = np.empty((N,))
    data[0] = _observe(x[:, 0], dt, gamma)

    m = np.empty((2, N))
    m[:, 0] = init_guess
    KF.initialize(m[:, 0], np.eye(2))

    for i in range(1, N):
        x[:, i] = continuous(x[:, i-1], sigma, dt, i).reshape((-1,))
        data[i] = _observe(x[:, i], dt, gamma)
        m[:, i] = KF.filter(data[i])[0].reshape((-1,))

    error = np.sqrt(np.sum((x-m)**2, axis=0))

    np.savez("filter_out", x=x, m=m, data=data, error=error)


def _load():
    f = np.load('filter_out.npz')
    return f['x'], f['m'], f['data'], f['error']


error = None


def show_result(figsize=20):
    global error
    x, m, data, error = _load()
    fig = Figure(size=figsize)
    fig.hline.opacities = [0.75]
    estline = fig.add_line([], [], 'go')

    def run(i, tail=True, sleep=0):
        if i == 0:
            fig.clear_tail()
        xi = x[:, i]
        mi = m[:, i]
        di = data[i]
        fig.update(xi, tail=tail, data=di)
        estline.x = [mi[0]]
        estline.y = [mi[1]]
        time.sleep(sleep)

    play = Play(value=0, min=0, max=1000, step=1)
    playbar = IntSlider(value=0, min=0, max=1000)
    jslink((play, 'value'), (playbar, 'value'))

    w = interactive(run, i=play,
                    tail=True, sleep=_spd_sld())
    display(HBox((fig.fig, VBox([w, playbar]))))


def plot_error():
    axes_options = {'x': {'label': 't'}, 'y': {'label': 'error'}}
    fig = plt.figure()
    plt.plot(np.arange(1000)*0.1, error, axes_options=axes_options)
    display(fig)


if __name__ == "__main__":
    show_result()
