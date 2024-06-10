import matplotlib.pyplot as plt
import numpy as np

font_size = 14

def fid_scaling_law_no_cfg():
    # data
    steps = np.array([50, 100, 200, 300,])
    loss_b = np.array([41.025, 33.442, 32.105, 32.196])
    loss_l = np.array([25.889, 24.654, 19.742, 19.070])
    loss_xl = np.array([19.820, 18.037, 14.772, 15.549])

    steps_ = np.array([50, 200, 300,])
    loss_xxl = np.array([17.195, 13.997, 14.648])
    loss_3b = np.array([16.431, 9.949, 9.380])
    # Plot
    plt.figure(figsize=(6, 4))

    plt.plot(steps, loss_b, 'o-', label='B', color='red')
    plt.plot(steps, loss_l, 'o-', label='L', color='orange')
    plt.plot(steps, loss_xl, 'o-', label='XL', color='green')
    plt.plot(steps_, loss_xxl, 'o-', label='XXL', color='blue')
    plt.plot(steps_, loss_3b, 'o-', label='3B', color='purple')

    plt.xlabel('Training Epochs', fontsize=font_size)
    plt.ylabel('FID', fontsize=font_size)
    # plt.grid(True)
    # plt.yscale('log')

    # Customize the plot to match the appearance of the provided figure
    plt.legend(loc='upper right', framealpha=0.5, fontsize=font_size, facecolor='white')

    # Customizing the x and y axis ticks (to match the example's steps)
    # plt.xticks(np.linspace(0, 800000, 5), ['0', '200K', '400K', '600K', '800K'])
    plt.yticks(np.arange(5, 50, step=5))

    # Show plot
    plt.tight_layout()
    plt.savefig('fid_scaling_law_no_cfg.png', dpi=600)



def fid_scaling_law_cfg():
    # data
    steps = np.array([50, 100, 200, 300,])
    loss_b_cfg = np.array([8.309, 7.256, 6.542, 6.249])
    loss_l_cfg = np.array([4.240, 3.705, 3.220, 3.075])
    loss_xl_cfg = np.array([3.420, 3.089, 2.617, 2.629])

    steps_ = np.array([50, 200, 300,])
    loss_xxl_cfg = np.array([2.893, 2.331, 2.340])
    loss_3b_cfg = np.array([2.611, 2.381, 2.329])
    # Plot
    plt.figure(figsize=(6, 4))

    plt.plot(steps, loss_b_cfg, 'o-', label='B', color='red')
    plt.plot(steps, loss_l_cfg, 'o-', label='L', color='orange')
    plt.plot(steps, loss_xl_cfg, 'o-', label='XL', color='green')
    plt.plot(steps_, loss_xxl_cfg, 'o-', label='XXL', color='blue')
    plt.plot(steps_, loss_3b_cfg, 'o-', label='3B', color='purple')

    plt.xlabel('Training Epochs', fontsize=font_size)
    plt.ylabel('FID', fontsize=font_size)
    # plt.grid(True)
    # plt.yscale('log')

    # Customize the plot to match the appearance of the provided figure
    plt.legend(loc='upper right', framealpha=0.5, fontsize=font_size, facecolor='white')

    # Customizing the x and y axis ticks (to match the example's steps)
    # plt.xticks(np.linspace(0, 800000, 5), ['0', '200K', '400K', '600K', '800K'])
    plt.yticks(np.arange(2, 9, step=1))

    # Show plot
    plt.tight_layout()
    plt.savefig('fid_scaling_law_cfg.png', dpi=600)



def sample_topk():
    # Data
    top_k = np.array([16384, 10000, 8000, 6000, 4000, 2000, 1000])
    fid_values = np.array([3.075, 3.369, 3.643, 3.969, 4.635, 5.998, 7.428])
    inception_scores = np.array([256.067, 265.222, 268.237, 270.159, 271.455, 267.278, 251.268])

    fig, ax1 = plt.subplots()
    # Create first y-axis
    ax1.set_xlabel('top-k', fontsize=font_size)
    ax1.set_ylabel('FID', color='teal', fontsize=font_size)
    ax1.plot(top_k, fid_values, 'o-', color='teal', label="FID")
    ax1.tick_params(axis='y', labelcolor='teal')
    ax1.tick_params(axis='x')

    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Inception Score', color='brown', fontsize=font_size)
    ax2.plot(top_k, inception_scores, 'o-', color='brown', label="Inception Score")
    ax2.tick_params(axis='y', labelcolor='brown')

    # Adding a legend
    fig.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), bbox_transform=ax1.transAxes, fontsize=font_size)

    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('effect_topk.png', dpi=600)



def sample_cfg():
    # Data
    cfg = np.array([1.5, 1.75, 2.00, 2.25])
    fid_values = np.array([4.743, 3.151, 3.075, 3.620])
    inception_scores = np.array([165.381, 214.152, 256.067, 291.695])

    plt.figure(figsize=(10, 4))
    fig, ax1 = plt.subplots()
    # Create first y-axis
    ax1.set_xlabel('cfg', fontsize=font_size)
    ax1.set_ylabel('FID', color='teal', fontsize=font_size)
    ax1.plot(cfg, fid_values, 'o-', color='teal', label="FID")
    ax1.tick_params(axis='y', labelcolor='teal')
    ax1.tick_params(axis='x')

    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Inception Score', color='brown', fontsize=font_size)
    ax2.plot(cfg, inception_scores, 'o-', color='brown', label="Inception Score")
    ax2.tick_params(axis='y', labelcolor='brown')

    # Adding a legend
    fig.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), bbox_transform=ax1.transAxes, fontsize=font_size)

    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('effect_cfg.png', dpi=600)



if __name__ == "__main__":
    fid_scaling_law_no_cfg()
    fid_scaling_law_cfg()
    sample_cfg()
    sample_topk()
