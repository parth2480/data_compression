import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 350  # number of samples
file = 'cleaned_data/C1_160313.txt'


def run(file, N):
    comp_size1 = []
    data = pd.read_csv(file)
    data.columns = ['Time', r'$B_x$', r'$B_y$', r'$B_z$']
    data = data.dropna()
    uncomp_size = N * 14 * 3
    return uncomp_size


def bit_count(b):
    h = []
    r = range(min(b), (max(b) + 1))
    r1 = np.arange(min(b), (max(b) + 1))
    for i in r:
        h.append(b.count(i))
    return h, r1


def bit_conv(a):
    bits = []
    for j in range(n - 1):
        if a[j] == 0:
            bits.append(1)
        else:
            bits.append(int(np.log2(abs(a[j])) + 1))
    return bits


data = pd.read_csv(file)
data.columns = ['Time', r'$B_x$', r'$B_y$', r'$B_z$']
data = data.dropna()

L = len(data['Time'])

trials = 500  # number of trials

max_xbits1 = []
max_ybits1 = []
max_zbits1 = []

max_xbits2 = []
max_ybits2 = []
max_zbits2 = []

comp_size1 = []
comp_size2 = []

uncomp_size = N * 14 * 3

time = np.linspace(0, N / 67, N)

for i in range(trials):
    r = int(np.random.uniform(0, L - N))

    chunk = data[r:r + N]
    data.astype({r'$B_y$': 'float64'})

    t = chunk['Time'].tolist()
    x_raw = chunk[r'$B_x$'].tolist()
    y_raw = chunk[r'$B_y$'].tolist()
    z_raw = chunk[r'$B_z$'].tolist()

    x_conv = [round(x / 0.0078125) for x in x_raw]
    y_conv = [round(y / 0.0078125) for y in y_raw]
    z_conv = [round(z / 0.0078125) for z in z_raw]

    global n
    n = len(x_conv)

    # Initial value difference

    x_diff = []
    y_diff = []
    z_diff = []

    for j in range(n):
        if j == 0:
            pass
        else:
            diff_x = x_conv[0] - x_conv[j]
            diff_y = y_conv[0] - y_conv[j]
            diff_z = z_conv[0] - z_conv[j]
            x_diff.append(diff_x)
            y_diff.append(diff_y)
            z_diff.append(diff_z)

    x_bits1 = bit_conv(x_diff)
    y_bits1 = bit_conv(y_diff)
    z_bits1 = bit_conv(z_diff)

    max_xbits1.append(max(x_bits1))
    max_ybits1.append(max(y_bits1))
    max_zbits1.append(max(z_bits1))

    # Previous value difference

    x_prev = []
    y_prev = []
    z_prev = []

    for j in range(n - 1):
        k = j + 1
        diff_x = x_conv[k] - x_conv[j]
        diff_y = y_conv[k] - y_conv[j]
        diff_z = z_conv[k] - z_conv[j]
        x_prev.append(diff_x)
        y_prev.append(diff_y)
        z_prev.append(diff_z)

    x_bits2 = bit_conv(x_prev)
    y_bits2 = bit_conv(y_prev)
    z_bits2 = bit_conv(z_prev)

    max_xbits2.append(max(x_bits2))
    max_ybits2.append(max(y_bits2))
    max_zbits2.append(max(z_bits2))

    total1 = (max(x_bits1) + max(y_bits1) + max(z_bits1))
    total2 = (max(x_bits2) + max(y_bits2) + max(z_bits2))

    comp_size1.append((total1 * N))
    comp_size2.append((total2 * N))


h_x1, r_x1 = bit_count(max_xbits1)
h_x2, r_x2 = bit_count(max_xbits2)
h_y1, r_y1 = bit_count(max_ybits1)
h_y2, r_y2 = bit_count(max_ybits2)
h_z1, r_z1 = bit_count(max_zbits1)
h_z2, r_z2 = bit_count(max_zbits2)

x_perc1 = [(14 - x) * (100 / 14) for x in max_xbits1]
y_perc1 = [(14 - y) * (100 / 14) for y in max_ybits1]
z_perc1 = [(14 - z) * (100 / 14) for z in max_zbits1]

x_perc2 = [(14 - x) * (100 / 14) for x in max_xbits2]
y_perc2 = [(14 - y) * (100 / 14) for y in max_ybits2]
z_perc2 = [(14 - z) * (100 / 14) for z in max_zbits2]


def plot1(single=False):
    if single:
        plt.bar(r_x1 - 0.3, h_x1, width=0.3, color='darkgrey', edgecolor='k')
        plt.bar(r_y1, h_y1, width=0.3, color='olive', edgecolor='k')
        plt.bar(r_z1 + 0.3, h_z1, width=0.3, color='goldenrod', edgecolor='k')
        plt.xlabel("No. of bits")
        plt.ylabel("No. of samples")
        plt.xticks(np.linspace(1, 12, 12))
        plt.legend(['x bits', 'y bits', 'z bits'])
        plt.title("Initial value coding", fontsize=10)
        plt.show()

        plt.bar(r_x2 - 0.3, h_x2, width=0.3, color='dimgrey', edgecolor='k')
        plt.bar(r_y2, h_y2, width=0.3, color='darkolivegreen', edgecolor='k')
        plt.bar(r_z2 + 0.3, h_z2, width=0.3, color='darkgoldenrod', edgecolor='k')
        plt.xlabel("No. of bits")
        plt.ylabel("No. of samples")
        plt.xticks(np.linspace(1, 9, 9))
        plt.legend(['x bits', 'y bits', 'z bits'])
        plt.title("Previous value coding", fontsize=10)
        plt.show()

        plt.hist([x_perc1, x_perc2], range=(0, 100), edgecolor='k', color=['darkgrey', 'dimgrey'], align='left')
        plt.xlabel("% Compression")
        plt.ylabel("No. of samples")
        plt.xticks(np.linspace(0, 100, 11))
        plt.xlim(0, 100)
        plt.legend(['Initial', 'Previous'])
        plt.title("x bits", fontsize=10)
        plt.show()

        plt.hist([y_perc1, y_perc2], range=(0, 100), edgecolor='k', color=['olive', 'darkolivegreen'], align='left')
        plt.xlabel("% Compression")
        plt.ylabel("No. of samples")
        plt.xticks(np.linspace(0, 100, 11))
        plt.xlim(0, 100)
        plt.legend(['Initial', 'Previous'])
        plt.title("y bits", fontsize=10)
        plt.show()

        plt.hist([z_perc1, z_perc2], range=(0, 100), edgecolor='k', color=['goldenrod', 'darkgoldenrod'], align='left')
        plt.xlabel("% Compression")
        plt.ylabel("No. of samples")
        plt.xticks(np.linspace(0, 100, 11))
        plt.xlim(0, 100)
        plt.legend(['Initial', 'Previous'])
        plt.title("z bits", fontsize=10)
        plt.show()
    else:
        f1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=[24, 12])
        ax1.bar(r_x1 - 0.3, h_x1, width=0.3, color='darkgrey', edgecolor='k')
        ax1.bar(r_y1, h_y1, width=0.3, color='olive', edgecolor='k')
        ax1.bar(r_z1 + 0.3, h_z1, width=0.3, color='goldenrod', edgecolor='k')
        ax1.set_xlabel("No. of bits")
        ax1.set_ylabel("No. of samples")
        ax1.set_xticks(np.linspace(1, 12, 12))
        ax1.legend(['x bits', 'y bits', 'z bits'])
        ax1.set_title("Initial value coding", fontsize=10)

        ax2.bar(r_x2 - 0.3, h_x2, width=0.3, color='dimgrey', edgecolor='k')
        ax2.bar(r_y2, h_y2, width=0.3, color='darkolivegreen', edgecolor='k')
        ax2.bar(r_z2 + 0.3, h_z2, width=0.3, color='darkgoldenrod', edgecolor='k')
        ax2.set_xlabel("No. of bits")
        ax2.set_ylabel("No. of samples")
        ax2.set_xticks(np.linspace(1, 8, 8))
        ax2.legend(['x bits', 'y bits', 'z bits'])
        ax2.set_title("Previous value coding", fontsize=10)

        m = np.linspace(1, trials, trials)

        ax3.axhline(uncomp_size, 0, trials, color='red')
        ax3.scatter(m, comp_size1, s=7.5, color='skyblue')
        ax3.axhline(np.mean(comp_size1), 0, trials, color='dodgerblue')
        ax3.scatter(m, comp_size2, s=7.5, color='darksalmon')
        ax3.axhline(np.mean(comp_size2), 0, trials, color='sienna')
        ax3.set_xlabel("Trials")
        ax3.set_ylabel("Size (in bits)")
        ax3.legend(['Uncompressed', 'Mean(Initial)', 'Mean(Previous)', 'Initial', 'Previous'])

        ax4.hist([x_perc1, x_perc2], range=(0, 100), edgecolor='k', color=['darkgrey', 'dimgrey'], align='left')
        ax4.set_xlabel("% Compression")
        ax4.set_ylabel("No. of samples")
        ax4.set_xticks(np.linspace(0, 100, 11))
        ax4.set_xlim(0, 100)
        ax4.legend(['Initial', 'Previous'])
        ax4.set_title("x bits", fontsize=10)

        ax5.hist([y_perc1, y_perc2], range=(0, 100), edgecolor='k', color=['olive', 'darkolivegreen'], align='left')
        ax5.set_xlabel("% Compression")
        ax5.set_ylabel("No. of samples")
        ax5.set_xticks(np.linspace(0, 100, 11))
        ax5.set_xlim(0, 100)
        ax5.legend(['Initial', 'Previous'])
        ax5.set_title("y bits", fontsize=10)

        ax6.hist([z_perc1, z_perc2], range=(0, 100), edgecolor='k', color=['goldenrod', 'darkgoldenrod'], align='left')
        ax6.set_xlabel("% Compression")
        ax6.set_ylabel("No. of samples")
        ax6.set_xticks(np.linspace(0, 100, 11))
        ax6.set_xlim(0, 100)
        ax6.legend(['Initial', 'Previous'])
        ax6.set_title("z bits", fontsize=10)
        plt.show()


def plot2(single=False):
    if single:
        plt.hist(max_xbits1, bins=(max(max_xbits1) - min(max_xbits1)), alpha=0.5, edgecolor='k', color='dimgrey')
        plt.hist(max_ybits1, bins=(max(max_ybits1) - min(max_ybits1)), alpha=0.5, edgecolor='k', color='darkolivegreen')
        plt.hist(max_zbits1, bins=(max(max_zbits1) - min(max_zbits1)), alpha=0.5, edgecolor='k', color='darkgoldenrod')
        plt.xlabel("No. of bits")
        plt.ylabel("No. of samples")
        plt.legend(['x bits', 'y bits', 'z bits'])
        plt.show()

        plt.hist(max_xbits2, bins=(max(max_xbits2) - min(max_xbits2)), alpha=0.5, edgecolor='k', color='dimgrey')
        plt.hist(max_ybits2, bins=(max(max_ybits2) - min(max_ybits2)), alpha=0.5, edgecolor='k', color='darkolivegreen')
        plt.hist(max_zbits2, bins=(max(max_zbits2) - min(max_zbits2)), alpha=0.5, edgecolor='k', color='darkgoldenrod')
        plt.xlabel("No. of bits")
        plt.ylabel("No. of samples")
        plt.legend(['x bits', 'y bits', 'z bits'])
        plt.show()

    else:
        f1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[16, 12])
        ax1.hist(max_xbits1, bins=(max(max_xbits1) - min(max_xbits1)), alpha=0.5, edgecolor='k', color='darkgrey')
        ax1.hist(max_ybits1, bins=(max(max_ybits1) - min(max_ybits1)), alpha=0.5, edgecolor='k', color='olive')
        ax1.hist(max_zbits1, bins=(max(max_zbits1) - min(max_zbits1)), alpha=0.5, edgecolor='k', color='orange')
        ax1.set_xlabel("No. of bits")
        ax1.set_ylabel("No. of samples")
        ax1.legend(['x bits', 'y bits', 'z bits'])

        ax2.hist(max_xbits2, bins=(max(max_xbits2) - min(max_xbits2)), alpha=0.5, edgecolor='k', color='dimgrey')
        ax2.hist(max_ybits2, bins=(max(max_ybits2) - min(max_ybits2)), alpha=0.5, edgecolor='k', color='darkolivegreen')
        ax2.hist(max_zbits2, bins=(max(max_zbits2) - min(max_zbits2)), alpha=0.5, edgecolor='k', color='darkgoldenrod')
        ax2.set_xlabel("No. of bits")
        ax2.set_ylabel("No. of samples")
        ax2.legend(['x bits', 'y bits', 'z bits'])

        ax3.hist([x_perc1, x_perc2], range=(0, 100), alpha=0.5, edgecolor='k', color=['darkgrey', 'dimgrey'])
        ax3.hist([y_perc1, y_perc2], range=(0, 100), alpha=0.5, edgecolor='k', color=['olive', 'darkolivegreen'])
        ax3.hist([z_perc1, z_perc2], range=(0, 100), alpha=0.5, edgecolor='k', color=['orange', 'darkgoldenrod'])
        ax3.set_xlabel("% bit compression")
        ax3.set_ylabel("No. of samples")
        ax3.legend(['x bits(Initial)', 'x bits(Previous)', 'y bits(Initial)', 'y bits(Previous)', 'z bits(Initial)', 'z bits(Previous)'])

        m = np.linspace(1, trials, trials)

        ax4.axhline(uncomp_size, 0, trials, color='red')
        ax4.scatter(m, comp_size1, s=7.5, color='skyblue')
        ax4.axhline(np.mean(comp_size1), 0, trials, color='dodgerblue')
        ax4.scatter(m, comp_size2, s=7.5, color='darksalmon')
        ax4.axhline(np.mean(comp_size2), 0, trials, color='sienna')
        ax4.set_xlabel("Trials")
        ax4.set_ylabel("Size (in bits)")
        ax4.legend(['Uncompressed', 'Mean(Initial)', 'Mean(Previous)', 'Initial', 'Previous'])
        plt.show()


plot1()
