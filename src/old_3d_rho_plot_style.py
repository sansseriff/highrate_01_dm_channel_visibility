# old 3d matrix plot style


real_matrix = np.real(rho)
imag_matrix = np.imag(rho)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = False

# Create a figure and axis objects for the real 3D bar plot

fig, (ax_real, ax_imag) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10,5))
ax_real.set_facecolor("none")
ax_imag.set_facecolor("none")
# plt.subplots( ncols=4, nrows=ceil(N/4), layout='constrained',
#                          figsize=(3.5 * 4, 3.5 * ceil(N/4)) )


# Set the x, y, and z coordinates for the real 3D bar plot
x = np.arange(real_matrix.shape[0])
y = np.arange(real_matrix.shape[1])
X, Y = np.meshgrid(x, y)
Z = real_matrix.ravel()

Z_abs_real = X

cmap = plt.get_cmap('plasma')
cmap_2 = plt.get_cmap('plasma')
# cmap_test = plt.get_cmap('plasma')


plot_real = ax_real.bar3d(X.ravel(), 
                          Y.ravel(), 
                          np.zeros_like(Z), 
                          0.8, 
                          0.8, 
                          Z, 
                          alpha=.4,
                          # color=cmap(Z_abs_real), 
                          edgecolor='black', 
                          linewidth=0.78,
                          cmap=cmap)

# Set the labels and title for the real 3D bar plot
ax_real.set_xlabel('Alice')
ax_real.set_ylabel('Bob')
ax_real.set_zlim(-1,1)
ax_real.set_zlabel('Real')
ax_real.set_title('Real 3D Bar Plot')

ax_real.set_xticks(x)
ax_real.set_xticklabels([r'$|EE\rangle$', r'$|EL\rangle$', r'$|LE\rangle$', r'$|LL\rangle$'])
ax_real.set_yticks(y)
ax_real.set_yticklabels([r'$|EE\rangle$', r'$|EL\rangle$', r'$|LE\rangle$', r'$|LL\rangle$'])



fig.colorbar(plot_real, ax=ax_real, location='bottom', cmap=cmap)

# Create a figure and axis objects for the imaginary 3D bar plot


# Set the x, y, and z coordinates for the imaginary 3D bar plot
Z = imag_matrix.ravel()
Z_abs_imag = Z

# Create the imaginary 3D bar plot
plot_imag = ax_imag.bar3d(X.ravel(), 
                          Y.ravel(), 
                          np.zeros_like(Z), 
                          0.8, 
                          0.8, 
                          Z, 
                          # color=cmap(Z_abs_imag), 
                          alpha=0.54, 
                          edgecolor='black', 
                          linewidth=0.78,
                          cmap=cmap_2)

# Set the labels and title for the imaginary 3D bar plot
ax_imag.set_xlabel('X')
ax_imag.set_ylabel('Y')
ax_imag.set_zlim(-.1,.1)
ax_imag.set_zlabel('Imaginary')
ax_imag.set_title('Imaginary 3D Bar Plot')

ax_imag.set_xticks(x)
ax_imag.set_xticklabels([r'$|EE\rangle$', r'$|EL\rangle$', r'$|LE\rangle$', r'$|LL\rangle$'])
ax_imag.set_yticks(y)
ax_imag.set_yticklabels([r'$|EE\rangle$', r'$|EL\rangle$', r'$|LE\rangle$', r'$|LL\rangle$'])

ax_imag.set_xlabel('Alice')
ax_imag.set_ylabel('Bob')

fig.colorbar(plot_imag, ax=ax_imag, location='bottom', cmap=cmap)


# fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')

# # Add a color bar to show the color ramp
# mappable = plt.cm.ScalarMappable(cmap=cmap)
# mappable.set_array(Z_abs_real)
# cbar = fig.colorbar(mappable, cax=ax_real)
# # fig.colorbar(im, ax=axs[0, 1], cax=axs[1, 0])

# Show the plots
# plt.subplots_adjust(left=-0, right=.5, top=.7, bottom=0)
# plt.margins(x=1,y=1)
# plt.gcf().subplots_adjust(left=0.15, right=0.5, bottom=0.15)
# plt.tight_layout()
plt.show()