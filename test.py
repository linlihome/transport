import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.use('TkAgg')
 
# plot double lines
def plot_double_lines(n, x, y1, y2, pic_name):
    # initialize plot parameters
    print('picture name: %s, len of data: %d' % (pic_name, n))
    plt.rcParams['figure.figsize'] = (10 * 16 / 9, 10)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08)

    # 对x和y1进行插值
    x_smooth = np.linspace(x.min(), x.max(), 50)
    y1_smooth = make_interp_spline(x, y1)(x_smooth)
    # plot curve 1

    plt.plot(x, y1, label='Score')
    
    # 对x和y2进行插值
    x_smooth = np.linspace(x.min(), x.max(), 50)
    y2_smooth = make_interp_spline(x, y2)(x_smooth)
    # plot curve 2
    plt.plot(x_smooth, y2_smooth, label='Similarity')
 
    # show the legend
    plt.legend()
 
    # show the picture
    plt.show()
 
 
if __name__ == '__main__':
    xs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y1s = np.array([8.0, 6.0, 5.7, 5.6, 5.2, 1.0, 0.8, 0.6])
    y2s = np.array([0.9, 0.8, 0.75, 0.41, 0.03, 0.01, 0.0, 1.0])
    plot_double_lines(len(xs), xs, y1s, y2s, 'Visualization of Linking Prediction')