# %% [markdown]
# # Imports

# %%
import numpy as np
import matplotlib.pyplot as plt



# %%
def box_char_func_np(x, x0, x1, y0, y1, z0, z1):
    """
    Determines if points in a NumPy array are within a specified 3D box.

    Parameters:
    x (np.ndarray): A 2D NumPy array of shape (n, 3) where each row represents a point in 3D space.
    x0 (float): The minimum x-coordinate of the box.
    x1 (float): The maximum x-coordinate of the box.
    y0 (float): The minimum y-coordinate of the box.
    y1 (float): The maximum y-coordinate of the box.
    z0 (float): The minimum z-coordinate of the box.
    z1 (float): The maximum z-coordinate of the box.

    Returns:
    np.ndarray: A 1D boolean array of length n where each element is True if the corresponding point is within the box, and False otherwise.
    """
    return np.logical_and.reduce((x[:, 0] >= x0, x[:, 0] <= x1, x[:, 1] >= y0, x[:, 1] <= y1, x[:, 2] >= z0, x[:, 2] <= z1))


# %%
def unif_concentration(x, a):
    """
    Generates a uniform concentration array.

    Parameters:
    x (numpy.ndarray): An array whose shape will be used to determine the size of the output array.
    a (float): The value to fill the output array with.

    Returns:
    numpy.ndarray: An array of the same length as the first dimension of `x`, filled with the value `a`.
    """
    return np.full(x.shape[0], a)

# %%
def clip(x, min_val, max_val):
    """
    Clips (limits) the values in an array.

    Parameters:
    x (numpy.ndarray): The array to clip.
    min_val (float): The minimum value to allow in the array.
    max_val (float): The maximum value to allow in the array.

    Returns:
    numpy.ndarray: A new array clipped so that all values are between `min_val` and `max_val`.
    """
    return np.clip(x, min_val, max_val)

# %%
def vertical_linear_gradient_dist(x, z_0, z_1, c_0, c_1):
    """
    Generates a linear gradient in the vertical direction.

    Parameters:
    x (numpy.ndarray): A 2D NumPy array of shape (n, 3) where each row represents a point in 3D space.
    z_0 (float): The minimum z-coordinate of the gradient.
    z_1 (float): The maximum z-coordinate of the gradient.
    c_0 (float): The value of the gradient at z=z_0.
    c_1 (float): The value of the gradient at z=z_1.

    Returns:
    numpy.ndarray: An array of length n with the gradient values.
    """
    _ = c_0 + (c_1 - c_0) * (x[:, 2] - z_0) / (z_1 - z_0)
    return np.clip(_, c_0, c_1)
    return _

# %%
def radial_linear_gradient_dist(x, r_0, r_1, c_0, c_1):
    """
    Generates a linear gradient in the radial direction.

    Parameters:
    x (numpy.ndarray): A 2D NumPy array of shape (n, 3) where each row represents a point in 3D space.
    r_0 (float): The minimum radial distance of the gradient.
    r_1 (float): The maximum radial distance of the gradient.
    c_0 (float): The value of the gradient at r=r_0.
    c_1 (float): The value of the gradient at r=r_1.

    Returns:
    numpy.ndarray: An array of length n with the gradient values.
    """
    r = np.linalg.norm(x[:, :2], axis=1)
    _ = c_0 + (c_1 - c_0) * (r - r_0) / (r_1 - r_0)
    return np.clip(_, min([c_0, c_1]), max([c_0, c_1]))

# %%
def y_linear_gradient(x, y_0, y_1, c_0, c_1):
    """
    Generates a linear gradient in the y direction.

    Parameters:
    x (numpy.ndarray): A 2D NumPy array of shape (n, 3) where each row represents a point in 3D space.
    y_0 (float): The minimum y-coordinate of the gradient.
    y_1 (float): The maximum y-coordinate of the gradient.
    c_0 (float): The value of the gradient at y=y_0.
    c_1 (float): The value of the gradient at y=y_1.

    Returns:
    numpy.ndarray: An array of length n with the gradient values.
    """
    _ = c_0 + (c_1 - c_0) * (x[:, 1] - y_0) / (y_1 - y_0)
    return np.clip(_, c_0, c_1)

# %% [markdown]
# # Slice inspection

# %%
def xy_slice_inspect(ax, conc_func, x0, x1, y0, y1, z=.5, n=100, v_min=0, v_max=1):
    """
    Generate and graph a vertical slice of a 3D function.

    Parameters:
    x0 (float): The minimum x-coordinate of the slice.
    x1 (float): The maximum x-coordinate of the slice.
    y0 (float): The minimum y-coordinate of the slice.
    y1 (float): The maximum y-coordinate of the slice.
    z (float): The z-coordinate of the slice.
    n (int): The number of points to sample in the x and y directions.

    Returns:
    numpy.ndarray: A 2D array of shape (n, n) where each element is the value of the function at that point in the slice.
    """

    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, z)
    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    conc = conc_func(points)
    conc = conc.reshape(n, n)

    contourf = ax.contourf(xx, yy, conc, levels=20, cmap='viridis', alpha=0.5, vmin=v_min, vmax=v_max)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    
    return contourf, conc

# %%
def yz_slice_inspect(ax, conc_func, y0, y1, z0, z1, x=.5, n=100, v_min=0, v_max=1):
    """
    Generate and graph a vertical slice of a 3D function.

    Parameters:
    y0 (float): The minimum y-coordinate of the slice.
    y1 (float): The maximum y-coordinate of the slice.
    z0 (float): The minimum z-coordinate of the slice.
    z1 (float): The maximum z-coordinate of the slice.
    x (float): The x-coordinate of the slice.
    n (int): The number of points to sample in the y and z directions.

    Returns:
    numpy.ndarray: A 2D array of shape (n, n) where each element is the value of the function at that point in the slice.
    """

    y = np.linspace(y0, y1, n)
    z = np.linspace(z0, z1, n)
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, x)
    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    conc = conc_func(points)
    conc = conc.reshape(n, n)

    contourf = ax.contourf(yy, zz, conc, levels=20, cmap='viridis', alpha=0.5, vmin=v_min, vmax=v_max)

    ax.set_xlim(y0, y1)
    ax.set_ylim(z0, z1)
    ax.set_aspect('equal')
    
    return contourf, conc

# %%
def xz_slice_inspect(ax, conc_func, x0, x1, z0, z1, y=.5, n=100, v_min=0, v_max=1):
    """
    Generate and graph a vertical slice of a 3D function.

    Parameters:
    x0 (float): The minimum x-coordinate of the slice.
    x1 (float): The maximum x-coordinate of the slice.
    z0 (float): The minimum z-coordinate of the slice.
    z1 (float): The maximum z-coordinate of the slice.
    y (float): The y-coordinate of the slice.
    n (int): The number of points to sample in the x and z directions.

    Returns:
    numpy.ndarray: A 2D array of shape (n, n) where each element is the value of the function at that point in the slice.
    """

    x = np.linspace(x0, x1, n)
    z = np.linspace(z0, z1, n)
    xx, zz = np.meshgrid(x, z)
    yy = np.full_like(xx, y)
    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    conc = conc_func(points)
    conc = conc.reshape(n, n)

    contourf = ax.contourf(xx, zz, conc, levels=20, cmap='viridis', alpha=0.5, vmin=v_min, vmax=v_max)

    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect('equal')
    
    return contourf, conc


# %%
# this keeps it to mcnp standards (sum must be 1)
def force_n_digits(x, n):
    # if x is less that 10^n, return 0000...x such that the length is n digits, else return x
    if x < 10**n:
        return f'{x:0{n}d}'
    return f'{x}'
# print(force_n_digits(100, 2))

def cut_bounds(extent, res):
    x0, x1, y0, y1, z0, z1 = extent
    x_res, y_res, z_res = res
    xs = np.linspace(x0, x1, x_res+1)
    ys = np.linspace(y0, y1, y_res+1)
    zs = np.linspace(z0, z1, z_res+1)

    return xs, ys, zs

def get_midpoints(sides, res, extent):
    x0, x1, y0, y1, z0, z1 = extent
    x_res, y_res, z_res = res
    xs, ys, zs = sides
    x_pad = (x1-x0)/x_res
    xs = xs+x_pad
    y_pad = (y1-y0)/y_res
    ys = ys+y_pad
    z_pad = (z1-z0)/z_res
    zs = zs+z_pad
    xx, yy, zz = np.meshgrid(xs[:-1], ys[:-1], zs[:-1])
    midpoints = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    return (xx, yy, zz), midpoints



def sample_section(f, midpoints, padding, n):
    # for every midpoint, sample the function f at n points in the extent
    x_pad, y_pad, z_pad = padding
    _X = [np.random.uniform(_[0]-x_pad, _[0]+x_pad, n) for _ in midpoints]
    _X = np.array(_X).flatten()
    _Y = [np.random.uniform(_[1]-y_pad, _[1]+y_pad, n) for _ in midpoints]
    _Y = np.array(_Y).flatten()
    _Z = [np.random.uniform(_[2]-z_pad, _[2]+z_pad, n) for _ in midpoints]
    _Z = np.array(_Z).flatten()
    points = np.column_stack((_X, _Y, _Z))
    elems = f(points)
    elems = np.reshape(elems, (len(midpoints), n, elems.shape[-1]))
    elems = np.mean(elems, axis=1)
    return elems

def fold128(text):
    """
    Folds a string to a maximum of 128 characters per line.

    Parameters:
    text (str): The input string to be folded.

    Returns:
    str: The folded string.
    """
    
    # print('text', text)
    new_lines = []
    lines = text.split(' ')
    
    new_line = ''
    i=0
    while i < len(lines):
        if len('\t'+new_line + lines[i] + ' ') < 128:
            new_line += lines[i] + ' '
            i += 1
        else:
            lines[i] = new_line
            new_line = ''
            new_lines.append(lines[i])
            i += 1
    # print('new_lines', new_lines)
    
    return '\n\t'.join(new_lines)

# %%
def make_mcnp(
        f, 
        extent, 
        res, 
        elem_labels, 
        density = -2.156, 
        density_map = None,
        x_fix = 0, 
        y_fix = 0, 
        z_fix=0, 
        z_mul = 1, 
        mw='w', 
        surface_header='', 
        surface_footer='', 
        mat_header='', 
        mat_footer='', 
        cell_header='', 
        cell_footer='',
        tally_header='',
        tally_footer='', 
        detector_tally_header='',
        detector_cell='',
        subsection_n=50
        ):
    """
    Generates MCNP input file components for a given geometry and material distribution.
    Parameters:
    f (function): A function that takes points and returns elements.
    extent (tuple): A tuple of six floats defining the extent of the geometry (x0, x1, y0, y1, z0, z1).
    res (tuple): A tuple of three integers defining the resolution in each dimension (x_res, y_res, z_res).
    elem_ids (list): A list of element IDs.
    surface_header (str, optional): A string to prepend to surface IDs. Defaults to ''.
    surface_footer (str, optional): A string to append to surface IDs. Defaults to ''.
    mat_header (str, optional): A string to prepend to material IDs. Defaults to ''.
    mat_footer (str, optional): A string to append to material IDs. Defaults to ''.
    cell_header (str, optional): A string to prepend to cell IDs. Defaults to ''.
    cell_footer (str, optional): A string to append to cell IDs. Defaults to ''.
    Returns:
    tuple: A tuple containing:
        - cells (str): The MCNP cell definitions.
        - cell_ids (list): A list of cell IDs.
        - surfaces (str): The MCNP surface definitions.
        - mats (str): The MCNP material definitions.
    """

    if mw == 'w':
        mw = -1
    elif mw == 'm':
        mw = 1

    x0, x1, y0, y1, z0, z1 = extent
    x_res, y_res, z_res = res
    xs, ys, zs = cut_bounds(extent, res)

    n = int(np.ceil(np.log10(max(x_res+1, y_res+1, z_res+1))))
    surfaces = ''
    x_ids = []
    y_ids = []
    z_ids = []
    for i, x in enumerate(xs):
        surface_id = f'{surface_header}1{force_n_digits(i, n)}{surface_footer}'
        x_ids.append(surface_id)
        surfaces += (f'{int(surface_id)} px {x-x_fix}\n')
    for i, y in enumerate(ys):
        surface_id = f'{surface_header}2{force_n_digits(i, n)}{surface_footer}'
        y_ids.append(surface_id)
        surfaces += (f'{int(surface_id)} py {y-y_fix}\n')
    for i, z in enumerate(zs):
        surface_id = f'{surface_header}3{force_n_digits(i, n)}{surface_footer}'
        z_ids.append(surface_id)
        surfaces += (f'{int(surface_id)} pz {z_mul*(z)-z_fix}\n')

    walls = (x_ids[0], x_ids[-1], y_ids[0], y_ids[-1], z_ids[0], z_ids[-1])

    XX, midpoints = get_midpoints(
        sides=(xs, ys, zs), res=res, extent=extent
    )
    xx, yy, zz = XX

    xx_index, yy_index, zz_index = np.meshgrid(x_ids[:-1], y_ids[:-1], z_ids[:-1])
    xx_index, yy_index, zz_index = xx_index.flatten(), yy_index.flatten(), zz_index.flatten()
    xl_ids, yl_ids, zl_ids = x_ids[1:], y_ids[1:], z_ids[1:]
    xxl_index, yyl_index, zzl_index = np.meshgrid(xl_ids, yl_ids, zl_ids)
    xxl_index, yyl_index, zzl_index = xxl_index.flatten(), yyl_index.flatten(), zzl_index.flatten()
    
    x_pad = (x1-x0)/res[0]
    y_pad = (y1-y0)/res[1]
    z_pad = (z1-z0)/res[2]
    pad = (x_pad, y_pad, z_pad)
    elems = sample_section(f, midpoints, pad, subsection_n)
    
    avg_sample = np.mean(elems, axis=0)
    mats = ''
    nn = int(np.ceil(np.log10(len(elems))))
    elem_ids = []
    for i, elem in enumerate(elems):
        elem_id = f'{mat_header}{force_n_digits(i, nn)}{mat_footer}'
        elem_ids.append(elem_id)
        mats += f'm{elem_id} '
        for id, e in zip(elem_labels, elem):
            if e == 0.0:
                continue
            else:
                mats += f'{id} {e*mw} '
        mats += '\n'

    cells = ''
    cell_ids = []

    tallies = ''
    tally_ids = []

    
    detector_tallies = [
        f'F{detector_tally_header}08:p {detector_cell}',
        f'E{detector_tally_header}08 0 1e-5 932i 8.4295',
        # f'T{detector_tally_header}08 0 150i 150',
        f'F{detector_tally_header}18:p {detector_cell}',
        f'E{detector_tally_header}18 0 1e-5 932i 8.4295',
        # f'T{detector_tally_header}18 0 150i 150',
        f'FT{detector_tally_header}18 GEB -0.026198 0.059551 -0.037176',
        f'*F{detector_tally_header}28:p {detector_cell}',
        f'E{detector_tally_header}28 0 1e-5 932i 8.4295',
        # f'T{detector_tally_header}28 0 150i 150',
        f'F{detector_tally_header}34:p {detector_cell}',
        f'CF{detector_tally_header}34:p', # put them here
        f'E{detector_tally_header}34 0 1e-5 932i 8.4295',
        # f'T{detector_tally_header}34 0 150i 150',
        f'F{detector_tally_header}36:p {detector_cell}',
        f'CF{detector_tally_header}36:p', # put them here
        f'E{detector_tally_header}36 0 1e-5 932i 8.4295',
        # f'T{detector_tally_header}36 0 150i 150',
        ]
    # for each line, if CF is in the line, add the corresponding CF to the list
    CF_indexs = [_ for _, e in enumerate(detector_tallies) if 'CF' in e]
    
    detector_tally_ids = []


    for i, e in enumerate(elem_ids):
        cell_id = f'{cell_header}{force_n_digits(i, nn)}{cell_footer}'
        cell_ids.append(cell_id)
        if density_map is None:
            cells += f'{cell_id} {e} {density} {xx_index[i]} -{xxl_index[i]} {yy_index[i]} -{yyl_index[i]} {z_mul*int(zz_index[i])} {z_mul*-int(zzl_index[i])} imp:n,p 1\n'
        else:
            density = np.multiply(density_map, elems[i])
            density = np.sum(density, axis=-1)
            cells += f'{cell_id} {e} {mw*density} {xx_index[i]} -{xxl_index[i]} {yy_index[i]} -{yyl_index[i]} {z_mul*int(zz_index[i])} {z_mul*-int(zzl_index[i])} imp:n,p 1\n'
        
        tally_id = f'{tally_header}{force_n_digits(i, nn)}{tally_footer}8'
        tally_ids.append(tally_id)
        tallies += f'F{tally_id}:p {cell_id} \n'
        tallies += f'e{tally_id} {cell_id} \n'


        tally_id = f'{tally_header}{force_n_digits(i, nn)}{tally_footer}4'
        tally_ids.append(tally_id)
        tallies += f'*F{tally_id}:p {cell_id} \n'
        tallies += f'e{tally_id} {cell_id} \n'

        tally_id = f'{tally_header}{force_n_digits(i, nn)}{tally_footer}06'
        tally_ids.append(tally_id)
        tallies += f'F{tally_id}:p {cell_id} \n'
        tallies += f'e{tally_id} {cell_id} \n'

        tally_id = f'{tally_header}{force_n_digits(i, nn)}{tally_footer}16'
        tally_ids.append(tally_id)
        tallies += f'+F{tally_id} {cell_id} \n'
        tallies += f'e{tally_id} {cell_id} \n'

    for cell_id in cell_ids:
        detector_tallies[8] += f' -{cell_id}'
        detector_tallies[11] += f' -{cell_id}'


    
    detector_tallies[8] = fold128(detector_tallies[8])
    detector_tallies[11] = fold128(detector_tallies[11])

    detector_tallies = '\n'.join(detector_tallies)+'\n'
    detector_tallies += 'T0 0 150i 150\n'

    detector_tallies = detector_tallies.split('\n')
    detector_tallies = [e for e in detector_tallies if e != '']
    detector_tallies = '\n'.join(detector_tallies)

    tallies = tallies.split('\n')
    tallies = [e for e in tallies if e != '']
    tallies = '\n'.join(tallies)

    elems = np.reshape(elems, (res[0], res[1], res[2], elems.shape[-1]))

    return cells, walls, surfaces, mats, avg_sample, elems, tallies, detector_tallies

