import os
import io
import PIL
from PIL import ImageFont
from PIL import ImageDraw

import torch
import numpy as np
import wandb
import imageio
import matplotlib.pyplot as plt
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
from sklearn.decomposition import PCA

from midi.analysis.rdkit_functions import Molecule


def visualize(path: str, molecules: list, num_molecules_to_visualize: int, log='graph', conformer2d=None,
              file_prefix='molecule'):
    """ molecules: List[Molecule]. """
    os.makedirs(path, exist_ok=True)
    if log == 'graph':
        pca = PCA(n_components=3)

    # visualize the final molecules
    if num_molecules_to_visualize == -1:
        num_molecules_to_visualize = len(molecules)
    if num_molecules_to_visualize > len(molecules):
        print(f"Shortening to {len(molecules)}")
        num_molecules_to_visualize = len(molecules)

    all_file_paths = []
    for i in range(num_molecules_to_visualize):
        mol = molecules[i]
        if log == 'graph':
            pos = mol.positions.cpu().numpy()
            if mol.positions.shape[0] > 2:
                pos = pca.fit_transform(pos)
            mol.positions = torch.from_numpy(pos).to(mol.atom_types.device)
        file_path = os.path.join(path, f'{file_prefix}{i}.png')
        plot_save_molecule(molecules[i], save_path=file_path, conformer2d=conformer2d)
        all_file_paths.append(file_path)

        if log is not None and wandb.run:
            wandb.log({log: wandb.Image(file_path)}, commit=True)

    return all_file_paths


def plot_save_molecule(mol, save_path, conformer2d=None):
    buffer = io.BytesIO()
    pil3d, max_dist = generatePIL3d(mol, buffer)
    new_im = PIL.Image.new('RGB', (600, 300), color='white')
    new_im.paste(pil3d, (0, 0, 300, 300))
    try:
        pil2d = generatePIL2d(mol.rdkit_mol, conformer2d)
        new_im.paste(pil2d, (300, 0, 600, 300))
    except ValueError:
        print("Value error in generate PIL2D. The ")
        return

    draw = ImageDraw.Draw(new_im)
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    try:        # This normally works but sometimes randomly crashes
        font = ImageFont.truetype(os.path.join(dir_path, "Arial.ttf"), 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((100, 15), f"3D view. Diam={max_dist:.1f}", font=font, fill='black')
    draw.text((420, 15), "2D view", font=font, fill='black')
    new_im.save(save_path, "PNG")
    buffer.close()


def generatePIL2d(mol, conformer2d=None):
    """ mol: RdKit molecule object
        conformer2d: n x 3 tensor defining the coordinates which should be used to plot (used for chains vis). """
    if conformer2d is None:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    if conformer2d is not None:
        conformer2d = conformer2d.double()
        for j, atom in enumerate(mol.GetAtoms()):
            x, y, z = conformer2d[j, 0].item(), conformer2d[j, 1].item(), conformer2d[j, 2].item()

            conf.SetAtomPosition(j, Point3D(x, y, z))
    return Draw.MolToImage(mol)


def visualize_chains(path, chain, atom_decoder, num_nodes):
    """ visualize the chain corresponding to one molecule"""
    RDLogger.DisableLog('rdApp.*')
    # convert graphs to the rdkit molecules

    pca = PCA(n_components=3)

    for i in range(chain.X.size(1)):        # Iterate over the molecules
        print(f'Visualizing chain {i}/{chain.X.size(1)}')
        result_path = os.path.join(path, f'chain_{i}')

        chain_atoms = chain.X[:, i][:, :num_nodes[i]].long()
        chain_charges = chain.charges[:, i][:, :num_nodes[i]].long()
        chain_bonds = chain.E[:, i][:, :num_nodes[i], :][:, :, :num_nodes[i]].long()
        chain_positions = chain.pos[:, i, :][:, :num_nodes[i]]

        # Transform the positions using PCA to align best to the final molecule
        if chain_positions[-1].shape[0] > 2:
            pca.fit(chain_positions[-1])
        mols = []
        for j in range(chain_atoms.shape[0]):
            pos = pca.transform(chain_positions[j]) if chain_positions[-1].shape[0] > 2 else chain_positions[j].numpy()
            mols.append(Molecule(atom_types=chain_atoms[j], charges=chain_charges[j], bond_types=chain_bonds[j],
                                 positions=torch.from_numpy(pos).to(chain_atoms.device),
                                 atom_decoder=atom_decoder))
        print("Molecule list generated.")

        # Extract the positions of the final 2d molecule
        last_mol = mols[-1].rdkit_mol
        AllChem.Compute2DCoords(last_mol)
        coords = []
        conf = last_mol.GetConformer()
        for k, atom in enumerate(last_mol.GetAtoms()):
            p = conf.GetAtomPosition(k)
            coords.append([p.x, p.y, p.z])
        conformer2d = torch.Tensor(coords)

        for frame in range(len(mols)):
            all_file_paths = visualize(result_path, mols, num_molecules_to_visualize=-1, log=None,
                                       conformer2d=conformer2d, file_prefix='frame')



        # Turn the frames into a gif
        imgs = [imageio.v3.imread(fn) for fn in all_file_paths]
        gif_path = os.path.join(os.path.dirname(path), f"{path.split('/')[-1]}_{i}.gif")
        print(f'Saving the gif at {gif_path}.')
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)

        if wandb.run:
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)
            # trainer.logger.experiment.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        print("Chain saved.")
    # draw grid image
    # try:
    #     img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
    #     img.save(os.path.join(path, f"{path.split('/')[-1]}_grid_image.png"))
    # except Chem.rdchem.KekulizeException:
    #     print("Can't kekulize molecule")
    # return mols


def plot_molecule3d(ax, positions, atom_types, edge_types, alpha, hex_bg_color, num_atom_types):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Normalize the positions for plotting
    max_x_dist = x.max() - x.min()
    max_y_dist = y.max() - y.min()
    max_z_dist = z.max() - z.min()
    max_dist = max(max_x_dist, max_y_dist, max_z_dist) / 1.8
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2
    z_center = (z.min() + z.max()) / 2
    x = (x - x_center) / max_dist
    y = (y - y_center) / max_dist
    z = (z - z_center) / max_dist

    radii = 0.4
    areas = 300 * (radii ** 2)
    if num_atom_types == 4:
        colormap = ['k', 'b', 'r', 'c']             # QM9 no H
    elif num_atom_types == 5:
        colormap = ['C7', 'k', 'b', 'r', 'c']
    elif num_atom_types == 16:
        colormap = ['C7', 'C0', 'k', 'b', 'r', 'c', 'C1', 'C2', 'C3', 'y', 'C5', 'C6', 'C8', 'C9', 'C10', 'C11']
    elif num_atom_types == 15:
        colormap = ['C0', 'k', 'b', 'r', 'c', 'C1', 'C2', 'C3', 'y', 'C5', 'C6', 'C8', 'C9', 'C10', 'C11']
    else:
        colormap = [f'C{a}' for a in range(num_atom_types)]

    colors = [colormap[a] for a in atom_types]
    for i in range(edge_types.shape[0]):
        for j in range(i + 1, edge_types.shape[1]):
            draw_edge = edge_types[i, j]
            if draw_edge > 0:
                ax.plot([x[i].cpu().numpy(), x[j].cpu().numpy()],
                        [y[i].cpu().numpy(), y[j].cpu().numpy()],
                        [z[i].cpu().numpy(), z[j].cpu().numpy()],
                        linewidth=1, c=hex_bg_color, alpha=alpha)

    ax.scatter(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), s=areas, alpha=0.9 * alpha, c=colors)
    return max_dist


def generatePIL3d(mol, buffer, bg='white', alpha=1.):
    atom_types = mol.atom_types
    edge_types = mol.bond_types
    positions = mol.positions
    num_atom_types = mol.num_atom_types
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#000000' #'#666666'

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='datalim')
    ax.view_init(elev=90, azim=-90)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    # max_value = positions.abs().max().item()
    axis_lim = 0.7
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    max_dist = plot_molecule3d(ax, positions, atom_types, edge_types, alpha, hex_bg_color, num_atom_types)

    plt.tight_layout()
    plt.savefig(buffer, format='png', pad_inches=0.0)
    pil_image = PIL.Image.open(buffer)
    plt.close()
    return pil_image, max_dist
