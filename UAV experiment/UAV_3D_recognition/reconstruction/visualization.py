
import numpy as np
from matplotlib import pyplot as plt

def plot_error_distribution(err_arr, bins=100, title='Data Graph', xlabel='Data', ylabel='Frequency'):
    norms = np.linalg.norm(err_arr, axis=0)
    
    plt.hist(norms, bins=bins)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()

def show_trajectory_3D(*X,title=None,color=True,line=False, size=20):
    fig = plt.figure(figsize=(12, 10))
    num = len(X)
    for i in range(num):
        ax = fig.add_subplot(1,num,i+1,projection='3d')
        
        if color:
            ax.scatter3D(X[i][0],X[i][1],X[i][2],c=np.arange(X[i].shape[1])*color, s=size)
        else:
            ax.scatter3D(X[i][0],X[i][1],X[i][2], s=size)

        if line:
            ax.plot(X[i][0],X[i][1],X[i][2])
        plt.xlabel('X')
        plt.ylabel('Y')
    if title:
        plt.suptitle(title)

    plt.show()

def plot_reconstruction_vs_ground_truth(reconstruction_result, ground_truth,scatter,time_stamp=None,linewidth=1,bbox_to_anchor=(1.2,1), output_file='Reconstructed-trajectory-result-scatter.pdf'):
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    ax = fig.add_subplot(111, projection='3d')

    if scatter==True:
        ax.scatter3D(reconstruction_result[0], reconstruction_result[1], reconstruction_result[2], c='r', s=3, label='Recognition')
        ax.scatter3D(ground_truth[0], ground_truth[1], ground_truth[2], c='g', s=1, label='Ground Truth', marker='^')
    else:
        x1 = reconstruction_result[0, :]
        x2 = ground_truth[0, :]
        y1 = reconstruction_result[1, :]
        y2 = ground_truth[1, :]
        z1 = reconstruction_result[2, :]
        z2 = ground_truth[2, :]
        diff_indices = np.where(np.diff(time_stamp) > 6)[0]
        time_segments = np.split(time_stamp, diff_indices + 1)
        x1_segments = np.split(x1, diff_indices + 1)
        x2_segments = np.split(x2, diff_indices + 1)
        y1_segments = np.split(y1, diff_indices + 1)
        y2_segments = np.split(y2, diff_indices + 1)
        z1_segments = np.split(z1, diff_indices + 1)
        z2_segments = np.split(z2, diff_indices + 1)
        for i in range(len(time_segments)):
            if len(x1_segments[i]) > 0 and len(y1_segments[i]) > 0 and len(z1_segments[i]) > 0:
                if i==0:
                    ax.plot(x1_segments[i], y1_segments[i], z1_segments[i], label='Recognition', color='r', linestyle='-',linewidth=linewidth)
                else:
                    ax.plot(x1_segments[i], y1_segments[i], z1_segments[i], color='r', linestyle='-',linewidth=linewidth)
            if len(x2_segments[i]) > 0 and len(y2_segments[i]) > 0 and len(z2_segments[i]) > 0:
                if i==0:
                    ax.plot(x2_segments[i], y2_segments[i], z2_segments[i], label='Ground-truth', color='g', linestyle='--',linewidth=linewidth)
                else:
                    ax.plot(x2_segments[i], y2_segments[i], z2_segments[i], color='g', linestyle='--',linewidth=linewidth)

    
    ax.scatter3D(reconstruction_result[0], reconstruction_result[1], 0, c='r', alpha=0.1, s=0.5)
    ax.scatter3D(ground_truth[0], ground_truth[1], 0, c='g', alpha=0.1, s=0.3, marker='^')

    for i in range(reconstruction_result.shape[1]):
        ax.plot([reconstruction_result[0, i], reconstruction_result[0, i]], 
                [reconstruction_result[1, i], reconstruction_result[1, i]], 
                [0, reconstruction_result[2, i]], color='r', alpha=0.02)
        ax.plot([ground_truth[0, i], ground_truth[0, i]], 
                [ground_truth[1, i], ground_truth[1, i]], 
                [0, ground_truth[2, i]], color='g', alpha=0.02)

    ax.set_xlabel('X(m)', fontsize=24, labelpad=20)
    ax.set_ylabel('Y(m)', fontsize=24, labelpad=25)
    ax.set_zlabel('Z(m)', fontsize=24, labelpad=25)

    ax.set_xlim(140,195)
    ax.set_ylim(140,200)
    ax.set_zlim(-10,300)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.legend(fontsize=18, loc='upper right', bbox_to_anchor=bbox_to_anchor, scatterpoints=5, handletextpad=0.5, markerscale=3, handlelength=2,frameon=False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    plt.savefig(output_file, format='pdf')

    plt.show()


def plot_trajectory_comparison(reconstruction_result, ground_truth,time_stamp, output_file='Reconstructed-trajectory-resultXYZ.pdf',
                               x_offset=-0.15, y_offset=1.05, font_size=24, font_weight='bold', font_family='serif'):
    x1 = reconstruction_result[0, :]
    x2 = ground_truth[0, :]
    y1 = reconstruction_result[1, :]
    y2 = ground_truth[1, :]
    z1 = reconstruction_result[2, :]
    z2 = ground_truth[2, :]


    diff_indices = np.where(np.diff(time_stamp) > 6)[0]

    time_segments = np.split(time_stamp, diff_indices + 1)
    x1_segments = np.split(x1, diff_indices + 1)
    x2_segments = np.split(x2, diff_indices + 1)
    y1_segments = np.split(y1, diff_indices + 1)
    y2_segments = np.split(y2, diff_indices + 1)
    z1_segments = np.split(z1, diff_indices + 1)
    z2_segments = np.split(z2, diff_indices + 1)

    plt.figure(figsize=(20, 10))

    plt.subplot(3, 1, 1)
    for i in range(len(time_segments)):
        if i == 0:
            plt.plot(time_segments[i], x1_segments[i], label='Recognition', color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], x2_segments[i], label='Ground-truth', color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)
        else:
            plt.plot(time_segments[i], x1_segments[i], color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], x2_segments[i], color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)

    plt.legend(loc='lower left', bbox_to_anchor=(-0.01, -0.08), fontsize=22, frameon=False)
    plt.ylabel('X(m)', fontsize=24)
    plt.ylim(110, 200)
    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.text(x_offset, y_offset, '(a)', transform=plt.gca().transAxes, fontsize=font_size, fontweight=font_weight, va='top', family=font_family)


    plt.subplot(3, 1, 2)
    for i in range(len(time_segments)):
        if i == 0:
            plt.plot(time_segments[i], y1_segments[i], label='Recognition', color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], y2_segments[i], label='Ground-truth', color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)
        else:
            plt.plot(time_segments[i], y1_segments[i], color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], y2_segments[i], color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)
    plt.ylabel('Y(m)', fontsize=24)
    plt.ylim(138, 192)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.text(x_offset, y_offset, '(b)', transform=plt.gca().transAxes, fontsize=font_size, fontweight=font_weight, va='top', family=font_family)

    plt.subplot(3, 1, 3)
    for i in range(len(time_segments)):
        if i == 0:
            plt.plot(time_segments[i], z1_segments[i], label='Recognition', color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], z2_segments[i], label='Ground-truth', color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)
        else:
            plt.plot(time_segments[i], z1_segments[i], color='red', marker='o', markersize=0,linestyle='-', linewidth=2.5)
            plt.plot(time_segments[i], z2_segments[i], color='green', marker='x', markersize=0,linestyle='--', linewidth=2.5)

    plt.xlabel('time series', fontsize=24)
    plt.ylabel('Z(m)', fontsize=24)
    plt.ylim(118, 190)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.text(x_offset, y_offset, '(c)', transform=plt.gca().transAxes, fontsize=font_size, fontweight=font_weight, va='top', family=font_family)

    plt.savefig(output_file, format='pdf')

    plt.show()