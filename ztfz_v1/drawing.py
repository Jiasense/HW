import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import platform

plt.rcParams['font.size'] = 15
if platform.system() == 'Windows':
    # plt.rcParams['font.family'] = ['simhei']
    plt.rcParams['font.sans-serif'] = ['simhei']
else:
    plt.rcParams['font.sans-serif'] = ['Hei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def to_array(obj):
    import torch
    import pandas as pd
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_numpy()
    else:
        return np.array(obj)


class Drawing:
    def __init__(self,
                 x_slider_min=-2,
                 y_slider_min=-2,
                 font_size=15,
                 ):
        plt.rcParams['font.size'] = font_size
        self.x_slider_min = x_slider_min
        self.y_slider_min = y_slider_min

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.twin_y_min = None
        self.twin_y_max = None

        self.x_start = None
        self.y_start = None
        self.twin_y_start = None

        self.x_window = None
        self.y_window = None
        self.twin_y_window = None

        self.use_twin = False

    def reset(self, figsize=(16, 9), dpi=100):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1)

        if self.use_twin:
            self.twin_ax = self.ax.twinx()

        self.fig.subplots_adjust(left=0.15, bottom=0.25)

        self.x_scale_ax = self.fig.add_axes([0.25, 0.05, 0.7, 0.05])
        self.x_start_ax = self.fig.add_axes([0.25, 0.1, 0.7, 0.05])
        w = 0.05 * figsize[1] / figsize[0]
        self.y_scale_ax = self.fig.add_axes([w, 0.25, w, 0.7])
        self.y_start_ax = self.fig.add_axes([w * 2, 0.25, w, 0.7])

        self.x_scale_slider = Slider(ax=self.x_scale_ax,
                                     label='sc',
                                     valmin=self.x_slider_min,
                                     valmax=0,
                                     valinit=0,
                                     )
        self.x_start_slider = Slider(ax=self.x_start_ax,
                                     label='st',
                                     valmin=0,
                                     valmax=1,
                                     valinit=0)

        if platform.system() == 'Windows':
            self.x_scale_slider.valtext.set_fontproperties(FontProperties(family='arial', size=15))
            self.x_start_slider.valtext.set_fontproperties(FontProperties(family='arial', size=15))
        else:
            self.x_scale_slider.valtext.set_fontproperties(FontProperties(family='DejaVu Sans', size=15))
            self.x_start_slider.valtext.set_fontproperties(FontProperties(family='DejaVu Sans', size=15))

        self.x_scale_slider.on_changed(self.update_x_scale)
        self.x_start_slider.on_changed(self.update_x_start)

        self.y_scale_slider = Slider(ax=self.y_scale_ax,
                                     label='sc',
                                     valmin=self.y_slider_min,
                                     valmax=0,
                                     valinit=0,
                                     orientation='vertical')
        self.y_start_slider = Slider(ax=self.y_start_ax,
                                     label='st',
                                     valmin=0,
                                     valmax=1,
                                     valinit=0,
                                     orientation='vertical',
                                     )

        if platform.system() == 'Windows':
            self.y_scale_slider.valtext.set_fontproperties(FontProperties(family='arial', size=15))
            self.y_start_slider.valtext.set_fontproperties(FontProperties(family='arial', size=15))
        else:
            self.y_scale_slider.valtext.set_fontproperties(FontProperties(family='DejaVu Sans', size=15))
            self.y_start_slider.valtext.set_fontproperties(FontProperties(family='DejaVu Sans', size=15))

        self.y_scale_slider.on_changed(self.update_y_scale)
        self.y_start_slider.on_changed(self.update_y_start)

        self.close_button_ax = self.fig.add_axes([w, 0.05, w, 0.05])
        self.close_button = Button(self.close_button_ax, 'X', color='red')
        self.close_button.on_clicked(self.click_close)

        self.left_button_ax = self.fig.add_axes([w * 3.5, 0.1 - 0.025, w, 0.05])
        self.left_button = Button(self.left_button_ax, '←', color='green')
        self.left_button.on_clicked(self.click_left)

        self.right_button_ax = self.fig.add_axes([w * 5.5, 0.1 - 0.025, w, 0.05])
        self.right_button = Button(self.right_button_ax, '→', color='green')
        self.right_button.on_clicked(self.click_right)

        self.up_button_ax = self.fig.add_axes([w * 4.5, 0.15 - 0.025, w, 0.05])
        self.up_button = Button(self.up_button_ax, '↑', color='green')
        self.up_button.on_clicked(self.click_up)

        self.down_button_ax = self.fig.add_axes([w * 4.5, 0.05 - 0.025, w, 0.05])
        self.down_button = Button(self.down_button_ax, '↓', color='green')
        self.down_button.on_clicked(self.click_down)

    def click_close(self, val):
        plt.close()

    def click_down(self, val):
        self.y_start -= self.y_window * 0.1
        self.y_start = max(self.y_min, self.y_start)
        self.update_y()
        self.y_start_slider.set_val((self.y_start - self.y_min) / (self.y_max - self.y_min))

    def click_up(self, val):
        self.y_start += self.y_window * 0.1
        self.y_start = min(self.y_start, self.y_max - self.y_window)
        self.update_y()
        self.y_start_slider.set_val((self.y_start - self.y_min) / (self.y_max - self.y_min))

    def click_left(self, val):
        self.x_start -= self.x_window * 0.1
        self.x_start = max(self.x_min, self.x_start)
        self.update_x()
        self.x_start_slider.set_val((self.x_start - self.x_min) / (self.x_max - self.x_min))

    def click_right(self, val):
        self.x_start += self.x_window * 0.1
        self.x_start = min(self.x_start, self.x_max - self.x_window)
        self.update_x()
        self.x_start_slider.set_val((self.x_start - self.x_min) / (self.x_max - self.x_min))

    def update_x(self):
        if self.x_start + self.x_window > self.x_max:
            a = self.x_max - self.x_window
            b = self.x_max
        else:
            a = self.x_start
            b = self.x_start + self.x_window

        self.ax.set_xlim(a, b)
        self.fig.canvas.draw()

    def update_y(self):
        if self.y_start + self.y_window > self.y_max:
            a = self.y_max - self.y_window
            b = self.y_max
        else:
            a = self.y_start
            b = self.y_start + self.y_window

        self.ax.set_ylim(a, b)

        if self.use_twin:
            if self.twin_y_start + self.twin_y_window > self.twin_y_max:
                a = self.twin_y_max - self.twin_y_window
                b = self.twin_y_max
            else:
                a = self.twin_y_start
                b = self.twin_y_start + self.twin_y_window

            self.twin_ax.set_ylim(a, b)

        self.fig.canvas.draw()

    def update_x_scale(self, val):
        value = 10 ** self.x_scale_slider.val
        self.x_window = (self.x_max - self.x_min) * value
        self.update_x()

    def update_x_start(self, val):
        value = self.x_start_slider.val
        self.x_start = (self.x_max - self.x_min) * value + self.x_min
        self.update_x()

    def update_y_scale(self, val):
        value = 10 ** self.y_scale_slider.val
        self.y_window = (self.y_max - self.y_min) * value

        if self.use_twin:
            self.twin_y_window = (self.twin_y_max - self.twin_y_min) * value

        self.update_y()

    def update_y_start(self, val):
        value = self.y_start_slider.val
        self.y_start = (self.y_max - self.y_min) * value + self.y_min
        if self.use_twin:
            self.twin_y_start = (self.twin_y_max - self.twin_y_min) * value + self.twin_y_min
        self.update_y()

    def init(self):
        # 在初始化画布后进行的额外操作
        pass

    @classmethod
    def run(cls,
            *ds,
            title='',
            save_path='',
            x_grid=False,
            ):
        cls().draw(*ds,
                   title=title,
                   save_path=save_path,
                   x_grid=x_grid
                   )

    def draw(self,
             *ds,
             title='',
             save_path='',
             x_grid=False,
             # use_twin=None,
             xticks=None,
             yticks=None,
             xlabel=None,
             ylabel=None,
             xlabelpad=None,
             ylabelpad=None,
             twin_yticks=None,
             twin_ylabel=None,
             date_formatter=None,
             ):

        self.use_twin = sum([each.get('axis', 0) for each in ds]) > 0

        self.reset()
        self.init()
        
        x_min = []
        x_max = []
        y_min = []
        y_max = []
        twin_y_min = []
        twin_y_max = []
        
        x = None

        for dic in ds:
            if 'y' not in dic:
                continue

            y = to_array(dic['y'])

            if self.use_twin:
                if len(ds) == 1:
                    y_min.append(y[:, 0].min())
                    y_max.append(y[:, 0].max())
                    if y.shape[1] > 2:
                        y_min.extend([*y[:, 2:].min(axis=0)])
                        y_max.extend([*y[:, 2:].max(axis=0)])
                    twin_y_min.append(y[:, 1].min())
                    twin_y_max.append(y[:, 1].max())
                else:
                    if dic.get('axis', 0) == 0:
                        y_min.append(y.min())
                        y_max.append(y.max())
                    else:
                        twin_y_min.append(y.min())
                        twin_y_max.append(y.max())

            else:
                y_min.append(y.min())
                y_max.append(y.max())

            args = []
            if 'x' not in dic:
                x_min.append(0)
                x_max.append(len(y) - 1)
            else:
                x = to_array(dic['x'])
                x_min.append(x.min())
                x_max.append(x.max())
                assert x.shape[0] == y.shape[0]
                args.append(x)

            kwargs = dict(marker=dic.get('marker'),
                          ms=dic.get('ms'),
                          mfc=dic.get('mfc'),
                          mec=dic.get('mec'),
                          linewidth=dic.get('linewidth'),
                          )

            if y.ndim == 2:
                for i in range(y.shape[1]):
                    ax = self.twin_ax if (self.use_twin and i == 1 and len(ds) == 1) \
                                         or (len(ds) >= 2 and dic.get('axis', 0) > 0) else self.ax
                    a = args.copy()
                    a.append(y[:, i])
                    kwargs['label'] = None if 'label' not in dic else dic['label'] + str(i)
                    kwargs['color'] = (np.random.rand(3, )
                                       if 'color' not in dic or not isinstance(dic['color'], (tuple, list))
                                       else dic['color'][i])
                    ax.plot(*a, **kwargs)
            else:
                ax = self.twin_ax if dic.get('axis', 0) > 0 else self.ax
                args.append(y)
                kwargs['label'] = dic.get('label')
                kwargs['color'] = dic.get('color')
                ax.plot(*args, **kwargs)

        self.x_min = min(x_min)
        self.x_max = max(x_max)
        self.y_min = min(y_min)
        self.y_max = max(y_max)

        self.x_start = self.x_min
        self.x_window = self.x_max - self.x_min
        self.y_start = self.y_min
        self.y_window = self.y_max - self.y_min

        self.ax.set_xlim(self.x_start, self.x_start + self.x_window)
        self.ax.set_ylim(self.y_start, self.y_start + self.y_window)

        if x_grid:
            self.ax.set_xticks(x if x is not None else range(self.x_min, self.x_max))
            self.ax.grid(axis='x', linestyle='--', alpha=0.7)

        if self.use_twin:
            self.twin_y_min = min(twin_y_min)
            self.twin_y_max = max(twin_y_max)
            self.twin_y_start = self.twin_y_min
            self.twin_y_window = self.twin_y_max - self.twin_y_min
            self.twin_ax.set_ylim(self.twin_y_start, self.twin_y_start + self.twin_y_window)

        if date_formatter is not None:
            self.ax.xaxis.set_major_formatter(date_formatter)

        if xticks is not None:
            self.ax.set_xticks(xticks)
        if xlabel is not None:
            self.ax.set_xlabel(xlabel,
                               labelpad=xlabelpad,
                               )
        if yticks is not None:
            self.ax.set_yticks(yticks)
        if ylabel is not None:
            self.ax.set_ylabel('\n'.join([s for s in ylabel]),
                               rotation=0,
                               labelpad=ylabelpad,
                               va='center',
                               )
        if twin_yticks is not None and self.use_twin:
            self.twin_ax.set_yticks(twin_yticks)
        if twin_ylabel is not None and self.use_twin:
            self.twin_ax.set_ylabel('\n'.join([s for s in twin_ylabel]),
                                    rotation=0,
                                    labelpad=ylabelpad,
                                    va='center',
                                    )
        if title:
            self.fig.suptitle(title)
        if any(['label' in dic for dic in ds]):
            self.fig.legend()

        if save_path:
            left = 1.25
            bottom = 1.6
            self.fig.savefig(save_path,
                             bbox_inches=Bbox.from_bounds(left, bottom, 16 - left, 9 - bottom))
            plt.close()
        else:
            plt.show()

    def __setattr__(self, name, value):
        if name in ['x_window', 'y_window'] and value == 0:
            super().__setattr__(name, 1.0)
        else:
            super().__setattr__(name, value)


if __name__ == '__main__':
    d = Drawing()
    d.draw({'y': np.sin(np.arange(-10, 10, 0.1)), 'color': 'orange', 'label': '0', 'axis': 0},
           {'y': np.cos(np.arange(-10, 10, 0.1)) * 0.5, 'color': 'red', 'label': '1', 'axis': 1},
           {'y': np.sin(np.arange(-10, 10, 0.1)) * 0.1, 'color': 'blue', 'label': '2', 'axis': 0},
           # use_twin=True,
           xlabel='t/s',
           # xticks=range(0, 200, 1),
           title='test0',
           ylabel='标签',
           yticks=np.arange(-1, 1, 0.1),
           # ylabelpad=10,
           # save_path='test.png',
           )

    # d.draw({'y': np.sin(np.arange(-20, 20, 0.1))}, title='test1')

    # import matplotlib.font_manager as font_manager
    # fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # for font in fonts:
    #     print(font)
