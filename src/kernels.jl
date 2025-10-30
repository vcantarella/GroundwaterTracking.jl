using KernelAbstractions
using GWGrids
using LinearAlgebra
"""
Define exit direction (left or right) based on current velocity of particle and the velocity at faces (run for each axis: x,y,z)
Parameters
----------
vleft: velocity at left face
vright: velocity at right face
v: current velocity

Returns
index (-1: exit at left face (negative axis direction),
        0: impossible exit at this axis
        1: exit at right face (positive axis direction)
-------

"""
@inline function exit_direction(vleft, vright, v)
    if (vleft > 0.0f0) & (vright > 0.0f0)
        return 1 # both faces positive velocity -> exit at right face
    elseif  (vleft < -0.0f0) & (vright < -0.0f0)
        return -1
    elseif  (vleft >= 0.0f0) & (vright <= -0.0f0)
        return 0
    else  # (v1 < 0) & (v2 > 0)
        if v > 0.0f0
            return 1
        elseif  v < -0.0f0
            return -1
        else
            return 0
        end
    end
end
"""
Calculates the time to reach the exit faces at each axis.

Parameters
----------
exit_ind: index calulated from exit_direction
gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
vleft: velocity at the lower face
vright: velocity at the upper face
v: current particle velocity
gv: velocity gradient in the cell
x: current particle coordinates
left_x: coordinates of lower left corner
right_x: coordinates of upper right corner

Returns
-------
dt: travel time at the current cell

"""
@inline function time_to_face(exit_ind, gradient_logic, vleft, vright, v, gv, x, left_x, right_x)
    if exit_ind == 0
        return Inf32
    elseif exit_ind == -1
        if gradient_logic
            vleft = abs(vleft)
            v = abs(v)
            return log(vleft/v) / gv
        else
            return (-1) * (x - left_x) / v
        end
    else  # exit_ind == 1
        if gradient_logic
            return log(vright/v) / gv
        else
            return (right_x - x) / v
        end
    end
end
@inline function argmin_cuda(dt_x, dt_y, dt_z)
    if dt_x <= dt_y
        if dt_x <= dt_z
            return 0
        else
            return 2
        end
    elseif dt_y <= dt_z
        return 1
    else
        return 2
    end
end
"""
Calculate the coordinates at the exit location in the cell
Parameters
----------
exit_ind: index calulated from exit_direction
gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
dt: calculated travel time at the cell
v1: velocity at the lower face
v2: velocity at the upper face
v: current particle velocity
gv: velocity gradient in the cell
x: current particle coordinates
left_x: coordinates of lower left corner
right_x: coordinates of upper right corner

Returns
-------
coords: exit coordinate at each axis
"""
@inline function exit_location(exit_ind, gradient_logic, dt, vleft, vright, v, gv, x, left_x, right_x, is_exit)
    if is_exit
        if exit_ind == 1
            x_new = right_x
        else
            x_new = left_x
        end
    elseif abs(v) > eps*abs(v)
        if not(gradient_logic)
            x_new = x + v * dt
        elseif exit_ind == 1
            x_new = left_x + (1 / gv) * (v * math.exp(gv * dt) - vleft)
        elseif exit_ind == -1
            x_new = right_x + (1 / gv) * (v * math.exp(gv * dt) - vright)
        else
            x_new = left_x + (1 / gv) * (v * math.exp(gv * dt) - vleft)
        end
    else
        x_new = x
    end
    return x_new
end

@kernel function travel_time!(
    result,
    @Const(initial_position),
    @Const(initial_cell),
    @Const(face_velocities),
    @Const(grid),
    # @Const(xedges),
    # @Const(yedges),
    # @Const(z_lowerface),
    # @Const(z_upperface),
    @Const(termination),
    @Const(accumulator),
)
    I = @index(Global)
    cell = initial_cell[I]
    position = initial_position[I]
    layer, row, col = cell
    x, y, z = position
    continue_tracking = true
    total_time = 0.0f0
    accm = 0.0f0
    count = 0
    # TODO: implement get_edges to a grid.
    x_edges, y_edges, z_lowerface, z_upperface = get_edges(grid)

    while continue_tracking
        # coordinates at left lower and upper right corners
        left_x = x_edges[col]
        right_x = x_edges[col + 1]
        left_y = y_edges[row]
        right_y = y_edges[row + 1]
        left_z = z_lowerface[layer]
        right_z = z_upperface[layer]
        relative_accm = accumulator[layer, row, col]
        # velocities at faces TODO: optimize memory access pattern for face velocities
        vleft_x = face_velocities[1, layer, row, col]
        vright_x = face_velocities[2, layer, row, col]
        vleft_y = face_velocities[3, layer, row, col]
        vright_y = face_velocities[4, layer, row, col]
        vleft_z = face_velocities[5, layer, row, col]
        vright_z = face_velocities[6, layer, row, col]
        # calculate gradients at current cell
        gv_x = (vright_x - vleft_x) / (right_x - left_x)
        gv_y = (vright_y - vleft_y) / (right_y - left_y)
        gv_z = (vright_z - vleft_z) / (right_z - left_z)
        # calculate velocity at current position
        vx = vleft_x + gv_x * (x - left_x)
        vy = vleft_y + gv_y * (y - left_y)
        vz = vleft_z + gv_z * (z - left_z)
        # determine exit directions
        exit_ind_x = exit_direction(vleft_x, vright_x, vx)
        exit_ind_y = exit_direction(vleft_y, vright_y, vy)
        exit_ind_z = exit_direction(vleft_z, vright_z, vz)
        if (exit_ind_x == 0) && (exit_ind_y == 0) && (exit_ind_z == 0)
            error("Particle is trapped at cell ($layer, $row, $col) at position ($x, $y, $z)")
            break
        end
        # calculate time to exit face in each axis
        dt_x = time_to_face(exit_ind_x, gv_x != 0.0f0, vleft_x, vright_x, vx, gv_x, x, left_x, right_x)
        dt_y = time_to_face(exit_ind_y, gv_y != 0.0f0, vleft_y, vright_y, vy, gv_y, y, left_y, right_y)
        dt_z = time_to_face(exit_ind_z, gv_z != 0.0f0, vleft_z, vright_z, vz, gv_z, z, left_z, right_z)
        # determine minimum time and exit axis
        dt_min = min(dt_x, dt_y, dt_z)
        exit_axis = argmin_cuda(dt_x, dt_y, dt_z)
        # update position
        exit_point_x = exit_location(exit_ind_x, gv_x != 0.0f0, dt_min, vleft_x, vright_x, vx, gv_x, x, left_x, right_x, exit_axis == 0)
        exit_point_y = exit_location(exit_ind_y, gv_y != 0.0f0, dt_min, vleft_y, vright_y, vy, gv_y, y, left_y, right_y, exit_axis == 1)
        exit_point_z = exit_location(exit_ind_z, gv_z != 0.0f0, dt_min, vleft_z, vright_z, vz, gv_z, z, left_z, right_z, exit_axis == 2)
        if exit_axis == 1
            col += 1
        elseif exit_axis == 2
            row += 1
        elseif exit_axis == 3
            layer += 1
        end

        #update dt
        total_time += dt_min
        accm += relative_accm * dt_min

        #check weather the particle has left the grid or reached termination condition
        if (col < 1) || (col > size(x_edges, 1) - 1) ||
             (row < 1) || (row > size(y_edges, 1) - 1) ||
              (layer < 1) || (layer > size(z_lowerface, 1))
            continue_tracking = false
        end

        if termination[layer, row, col] == true
            continue_tracking = false
        end

        #update position
        x = exit_point_x
        y = exit_point_y
        z = exit_point_z
        count += 1
    end
    result[I, 1] = total_time
    result[I, 2] = accm
    result[I, 3] = count
end
