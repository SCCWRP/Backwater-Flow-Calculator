# Dependencies
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from bisect import bisect_left

# Hard-coded project parameters - Need to read from template
surface_area_sf = 1262
dim_width_ft = 50
dim_length_ft = 25
side_slope = .33
radius_in = 6
length_ft = 27.583
nMannings = np.arange(0.009, 0.033, 0.002)
slope = 0.002
gravity_SI = 9.81
composite_volume_total_L = 3

# Hard-coded program parameters, including conversion factors
sf_to_m2 = 0.0929
in_to_m = 0.0254
ft_to_m = 0.3048
min_to_sec = 60.0
cfs_to_cms = 0.0283
Q_guess_factor = 2
Q_guess_interval = 1000

# Use pandas to read data template
data = pd.read_excel(r'C:/Users/edwardt/SCCWRP/SMC BMP Projects - Regional monitoring network/Flow Calculations from Depth - M1 Backwater Curve/Flow_calculator_repo/FlowCalc_Template.xlsx', sheet_name='Caltrans 1194 - Dec Check')
df_depth = pd.DataFrame(data, columns=['Elapsed Time Depth (min)', 'y1 (in)', 'y2 (in)'])
df_flow = pd.DataFrame(data, columns=['Elapsed Time Flow (min)', 'Exfiltration (cfs)'])
# df_sample = pd.DataFrame(data, columns=['Elapsed Time Sample (min)'])

# Strip out NA's from dataframes, store as 1D-list variables
y1_depth = df_depth['y1 (in)'][df_depth['y1 (in)'].notna()]
y2_depth = df_depth['y2 (in)'][df_depth['y2 (in)'].notna()]
y_time = df_depth['Elapsed Time Depth (min)'][df_depth['Elapsed Time Depth (min)'].notna()]
Qex_time = df_flow['Elapsed Time Flow (min)'][df_flow['Elapsed Time Flow (min)'].notna()]
Qex_flow = df_flow['Exfiltration (cfs)'][df_flow['Exfiltration (cfs)'].notna()]
# sample_time = df_sample['Elapsed Time Sample (min)'][df_sample['Elapsed Time Sample (min)'].notna()]


def settings_SI():
    global surface_area_m2, radius_m, length_m, dim_width_m, dim_length_m, zz_drop_m, area_full_m2, wp_full_m, rh_full_m

    # Convert units to SI
    surface_area_m2 = surface_area_sf * sf_to_m2
    radius_m = radius_in * in_to_m
    length_m = length_ft * ft_to_m
    dim_width_m = dim_width_ft * ft_to_m
    dim_length_m = dim_length_ft * ft_to_m

    zz_drop_m = slope * length_m
    area_full_m2 = np.pi * radius_m**2
    wp_full_m = np.pi * radius_m * 2
    rh_full_m = area_full_m2/wp_full_m

    # Convert min -> sec, in -> m, cfs -> cms
    for ii in range(len(y_time)):
        y_time[ii] = y_time[ii] * min_to_sec
        y1_depth[ii] = y1_depth[ii] * cfs_to_cms
        y2_depth[ii] = y2_depth[ii] * cfs_to_cms

        # If instrument-recorded depths are negative, set them to zero
        if y1_depth[ii] < 0:
            y1_depth[ii] = 0
        if y2_depth[ii] < 0:
            y2_depth[ii] = 0

    for ii in range(len(Qex_time)):
        Qex_time[ii] = float(Qex_time[ii] * min_to_sec)
        Qex_flow[ii] = Qex_flow[ii] * cfs_to_cms

    # for ii in range(len(sample_time)):
    #     sample_time[ii] = sample_time[ii] * min_to_sec

    return 


def mannings_eq_flow(y1, y2, nM):
    """
    Flow from Manning's equation is calculated from average depth to use as a seed for the iterative solution

    Assumptions: the pipe slope is the same as the friction slope, this is enforced by taking the average area and hydraulic radius
    """

    # Calculate average flow depth in pipe
    y_ave = np.average([y1, y2])

    # Calculate the internal angle from flow depth
    theta_ave = theta_calc(y_ave)

    # Calculate the cross-sectional flow area from flow depth and internal angle
    area_ave = flow_area(y1, theta_ave)

    # Calculate the wetted perimeter from flow depth and internal angle
    wp_ave = wetted_perimeter(y1, theta_ave)

    # Calculate the hydraulic radius from cross-sectional flow area and wetted perimeter
    rh_ave = area_ave/wp_ave
    
    # Calculate flowrate from Manning's Eq - assumes friction slope is represented by pipe slope
    Q_mannings = (1/nM) * area_ave * slope**(1/2) * rh_ave**(2/3)

    # Return an instantaneous flow rate from flow depth, Manning's roughness
    return Q_mannings


def Energy_eq_check(y1, y2, nM, Q_mannings_guess):
    """
    This subroutine uses an iterative approach to solve the Energy equation for flow under gradually varying flow conditions (i.e. y1 != y2)

    Flowrate guesses are seeded using Manning's equation, the flowrate value which minimizes the Energy eq residual is considered the instantaneous flow rate
    """

    # The Energy eq residual is set to infinity at the start
    old_energy_residual = np.inf

    # The target flowrate is set to zero at the start
    qq_target = 0

    # The range of flowrates is bounded by Manning's eq result and hardcoded range parameters
    Q_low = max(Q_mannings_guess * (1 / Q_guess_factor), 1e-10)
    Q_high = max(Q_mannings_guess * Q_guess_factor, 1e-9)
    Q_interval = (Q_high - Q_low) / Q_guess_interval
    potential_flowrates = np.arange(Q_low, Q_high, Q_interval)

    # Pipe-flow geometry parameters calculated here
    theta1 = theta_calc(y1)
    theta2 = theta_calc(y2)
    area1 = flow_area(y1, theta1)
    area2 = flow_area(y2, theta2)
    wp1 = wetted_perimeter(y1, theta1)
    wp2 = wetted_perimeter(y2, theta2)
    rh1 = area1/wp1
    rh2 = area2/wp2

    # Average geometry parameters are used to estimate friction slope in Manning's eq
    area_ave = np.average([area1, area2])
    rh_ave = np.average([rh1, rh2])
    
    """The energy equation is composed of 4 head terms: Velocity1, Velocity2, Friction Loss, Elevation
        of which 3 are functions of flow rate: Velocity1, Velocity2, Friction Loss

    Velocity1(Q) - Velocity2(Q) - Friction Loss(Q) - Elevation = residual

    ^The flowrate that minimizes the residual is considered the instantaneous flow rate
    """
    # Elevation head is the free surface elevation difference between y2 and y1 (from raw data, does not depend on Q)
    elev_head_diff = y2 - y1 - zz_drop_m

    # This for loop solves for the energy equation residual
    # HACK - should vectorize this loop
    for qq in potential_flowrates:

        # Calculate Velocity and Friction heads

        # Velocity head uses the Continuity equation such that V = Q/A
        vel_head1 = qq**2 / (2 * gravity_SI * area1**2)
        vel_head2 = qq**2 / (2 * gravity_SI * area2**2)

        # Friction head uses Manning's equation solved for friction slope times the pipe length
        fric_head = (qq**2 * nM**2 * length_m) / (area_ave**2 * rh_ave**(4/3))

        # Calculate the new residual, update the flowrate guess & old residual if new value is smaller
        new_energy_residual = np.abs(vel_head1 - vel_head2 - fric_head - elev_head_diff)
        if new_energy_residual < old_energy_residual:
            old_energy_residual = new_energy_residual
            qq_target = qq

    # Flowrate Q such that residual is minimized
    return qq_target


# This doesn't work, ignore for now
def Gradually_varied_pipe_flow(y1_depth, y2_depth):
    Qin_GVPF = np.zeros(len(y_time))

    area1 = np.empty(len(y_time))
    area2 = np.empty(len(y_time))
    rh1 = np.empty(len(y_time))
    rh2 = np.empty(len(y_time))
    area_ave = np.empty(len(y_time))
    rh_ave = np.empty(len(y_time))

    rel_elevation_head = np.zeros(len(y_time))
    rel_velocity_head = np.zeros(len(y_time))
    friction_head = np.zeros(len(y_time))

    radius_m_arr = np.full(len(y_time), radius_m)
    area_full_m2_arr = np.full(len(y_time), area_full_m2)
    rh_full_m_arr = np.full(len(y_time), rh_full_m)

    # for yy in range(len(y1_depth)):
    for yy in range(50):
        y1 = y1_depth[yy]
        y2 = y2_depth[yy]

        theta1 = theta_calc(y1)
        theta2 = theta_calc(y2)
        area1[yy] = flow_area(y1, theta1)
        area2[yy] = flow_area(y2, theta2)

        wp1 = wetted_perimeter(y1, theta1)
        wp2 = wetted_perimeter(y2, theta2)
        rh1[yy] = area1[yy]/wp1
        rh2[yy] = area2[yy]/wp2
        area_ave[yy] = np.average([area1[yy], area2[yy]])
        rh_ave[yy] = np.average([rh1[yy], rh2[yy]])

        rel_elevation_head[yy] = y2 - y1 - zz_drop_m
        rel_velocity_head[yy] = 1 / (2 * gravity_SI * (area1[yy]**2 - area2[yy]**2))
        friction_head[yy] = (nMannings**2 * length_m) / (area_ave[yy]**2 * rh_ave[yy]**(4/3))
        inv_terms = 1 / (rel_velocity_head[yy] - friction_head[yy])
        print(area_ave[yy], rh_ave[yy], friction_head[yy])
        try:
            Qin_GVPF[yy] = np.sqrt(rel_elevation_head[yy] * inv_terms)
        except RuntimeWarning:
            Qin_GVPF[yy] = 0
            print("Q not possible")

    # fig, axs = plt.subplots(2)
    # fig.suptitle('Elevation, Velocity, Friction Head vs Time')

    # axs[0].plot(y_time[0:263], rel_elevation_head[0:263], color='Blue')
    # axs[0].plot(y_time[0:263], rel_velocity_head[0:263], color='Red')
    # axs[0].plot(y_time[0:263], friction_head[0:263], color='Green')
    # axs[0].set(ylabel="Head (m)")

    # axs[1].plot(y_time[0:263], y1_depth[0:263], color='Blue')
    # axs[1].plot(y_time[0:263], y2_depth[0:263], color='Red')
    # axs[1].plot(y_time[0:263], radius_m_arr[0:263], color='Black')
    # axs[1].set(xlabel="Time since start (s)", ylabel="Flow Depth (m)")

    # # axs[2].plot(y_time[0:263], rh1[0:263], color='Blue')
    # # axs[2].plot(y_time[0:263], rh2[0:263], color='Red')
    # # axs[2].plot(y_time[0:263], rh_full_m_arr[0:263], color='Black')
    # # axs[2].set(xlabel="Time since start (s)", ylabel="Hydraulic Radius (m)")
    # plt.show()
    # quit()

    return Qin_GVPF


def theta_calc(flow_depth):
    """
    This subroutine calculates the internal angle from pipe center to free surface.

    The conditions describe adjustments to the internal angle equation based on whether the free surface is above or below the pipe center
    """

    # Default reference depth is zero
    reference_depth = 0

    # Diameter from Radius
    diamter_m = 2 * radius_m

    # If the flow depth is between zero and the radius, reference depth is the flow depth
    if 0 <= flow_depth < radius_m:
        reference_depth = flow_depth

    # Else-if the flow depth is between the radius and diameter (i.e. above halfway full) the reference depth is the diameter minus flow depth
    elif radius_m <= flow_depth < diamter_m:
        reference_depth = diamter_m - flow_depth

    # Else-if the flow depth is greater than the diameter, the reference depth doesn't mean anything
    elif flow_depth >= diamter_m:
        "Depth is greater than the pipe, area full"

    # Else, the depth is negative and needs to be checked
    else:
        "Check the depth condition, something has gone wrong"

    # The internal angle is calculated from the reference depth and pipe radius
    try:
        theta = 2*np.arccos(1 - reference_depth/radius_m)
    except:
        theta = 0
    return theta


def flow_area(depth, theta):
    global full_area_flag
    """
    This subroutine calculates the cross-sectional flow area from flow depth and internal angle

    Equation used depends on ratio of flow depth/pipe radius
    """

    # Full area flag is used to add an orifice loss term to energy equation if True
    full_area_flag = False

    # Diameter from Radius
    diamter_m = 2 * radius_m

    # Small depth correction - uses rectangular approximation of area for flows 
    small_depth_threshold = radius_m * 0.001
    if 0 <= depth < small_depth_threshold:
        area = radius_m * depth

    # Else-if flow depth is less than the radius, calculate area directly
    elif small_depth_threshold <= depth < radius_m:
        area = radius_m**2*(theta - np.sin(theta))

    # Else-if flow depth is less than the diameter, calculate area as full - empty area
    elif radius_m <= depth < diamter_m:
        area = area_full_m2 - radius_m**2*(theta - np.sin(theta))

    # Else-if flow depth is greater than the diameter, the area is full and the full_area_flag is turned on
    elif depth >= diamter_m:
        area = area_full_m2
        full_area_flag = True
    
    # Error condition, stop
    else:
        area = 0
        print("Check the depth condition, something has gone wrong")
        print(depth)
        quit()
    return area


def wetted_perimeter(depth, theta):
    """
    This subroutine calculates the wetted perimeter from flow depth and internal angle

    Equation used depends on ratio of flow depth/pipe radius
    """

    # Diameter from Radius
    diamter_m = 2 * radius_m

    # Wetted perimeter if pipe full
    full_perimeter = diamter_m * np.pi

    # If less than half full
    if 0 <= depth < radius_m:
        wp = full_perimeter - radius_m * theta

    # If more than half full
    elif radius_m <= depth < diamter_m:
        wp = radius_m * theta

    # If more than completely full
    elif depth >= diamter_m:
        wp = full_perimeter

    # Anything else means something is wrong with the depth condition
    else:
        wp = 0
        "Check the depth condition, something has gone wrong"
        print(depth)
        quit()
    return wp


def SandFilter_CV(y2_depth, y_time, Q_exfil, Q_exfil_time):
    """
    This subroutine uses the Storage equation to calculate inflow as a function of the change of measured y2 depth and outflow

    Args: list arguments
    """

    # Initialize the inflow array
    Qin_SFSH = np.zeros(len(y_time))
    Qin_SFSH[0] = 0
    
    # Manage the outflow timeseries so that it matches the depth timeseries
    Q_exfil_short = timeseries_mask_to_depth(y_time, Q_exfil, Q_exfil_time)

    # Storage equation needs the change in depth with time
    for ii in range(len(y2_depth)-1):
        # Depth derivative with time, trapezoidal approximation
        delta_y = y2_depth[ii+1] - y2_depth[ii]
        delta_t = y_time[ii+1] - y_time[ii]
        dydt = delta_y/delta_t

        # Update the filter surface area given new depth
        surface_area_new = surface_area(y2_depth[ii+1])

        # Calculate the change in storage volume between the free surface and perforated pipe outfall
        d_storage = dydt*surface_area_new

        # Calculate the inflow as the change 
        Qin_SFSH[ii+1] = d_storage + Q_exfil_short[ii+1]

    # Return the inflow and (adjusted) outflow timeseries.  These are the same shape as y_time
    return Qin_SFSH, Q_exfil_short


def surface_area(y2):
    """
    This subroutine is used to calculate the surface area as a function of water depth above Sand Filter
    """

    # Use trapezoidal approximation of BMP geometry to calculate new surface area
    new_sa = (dim_width_m + 2*y2*side_slope) * (dim_length_m + 2*y2*side_slope)

    # The water free surface area is returned as the Sand Filter surface area + the trapezoidal prism contribution
    return new_sa


def timeseries_mask_to_depth(y_time, Qex_flow, Qex_time):
    """
    This subroutine is used to align the outflow timeseries with the depth timeseries so they can be used
    in the Storage equation more easily
    """
    # Initialize the masked outflow list
    Q_exfil_DepthMask = []

    y_time = list(y_time)
    Qex_time = list(Qex_time)

    # When the outflow timeseries elapsed time is aligned with the depth timeseries elapsed time, include value in the masked outflow timeseries
    for ii in range(len(Qex_time)):
        if Qex_time[ii] in y_time:
            Q_exfil_DepthMask.append(Qex_flow[ii])
    
    return Q_exfil_DepthMask


def volume_calc(y_series, x_series):
    """
    This subroutine calculates the trapezoidal approximation of the area under the curve
    """

    volume = np.trapz(y_series, x_series)
    return volume


def match_sample_to_inflow(sample_timeseries, y_time, Qinflow):
    Q_sample = []
    sample_timeseries = list(sample_timeseries)
    y_time = list(y_time)
    Qinflow = list(Qinflow)

    # When the outflow timeseries elapsed time is aligned with the depth timeseries elapsed time, include value in the masked outflow timeseries
    for ii in range(len(y_time)):
        if y_time[ii] in sample_timeseries:
            Q_sample.append(Qinflow[ii])

    return Q_sample


def volume_alloc(sample_timeseries, y_time, Qinflow, Comp_Vol):

    sample_timeseries = list(sample_timeseries)
    sample_volumes = []
    y_time = list(y_time)
    Qinflow = list(Qinflow)
    sample_timeseries = [0] + sample_timeseries + [y_time[-1]]

    sample_leftedge = 0
    time_index_left = 0
    sample_rightedge = np.average([sample_timeseries[1], sample_timeseries[2]])
    time_index_right = y_time.index(take_closest(y_time, sample_rightedge))

    Qinflow_snip = Qinflow[time_index_left:time_index_right+1]
    y_time_snip = y_time[time_index_left:time_index_right+1]
    volume_snip = volume_calc(Qinflow_snip, y_time_snip)
    sample_volumes.append(volume_snip)

    for ii in range(2, len(sample_timeseries)-1):

        sample_leftedge = np.average([sample_timeseries[ii-1], sample_timeseries[ii]])
        time_index_left = y_time.index(take_closest(y_time, sample_leftedge))

        sample_rightedge = np.average([sample_timeseries[ii], sample_timeseries[ii+1]])
        time_index_right = y_time.index(take_closest(y_time, sample_rightedge))
        
        Qinflow_snip = Qinflow[time_index_left:time_index_right+1]
        y_time_snip = y_time[time_index_left:time_index_right+1]
        volume_snip = volume_calc(Qinflow_snip, y_time_snip)
        sample_volumes.append(volume_snip)


    sample_leftedge = np.average([sample_timeseries[-2], sample_timeseries[-1]])
    time_index_left = y_time.index(take_closest(y_time, sample_leftedge))

    Qinflow_snip = Qinflow[time_index_left:]
    y_time_snip = y_time[time_index_left:]
    volume_snip = volume_calc(Qinflow_snip, y_time_snip)
    sample_volumes.append(volume_snip)

    vol_total = sum(sample_volumes)
    volume_allocated = np.empty(len(sample_volumes))
    for ss in range(len(sample_volumes)):
        volume_allocated[ss] = Comp_Vol * sample_volumes[ss] / vol_total
    return volume_allocated


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def determine_hydrographs(y1_depth, y2_depth, y_time, Qex_flow, Qex_time, nMannings, sample_time):
    settings_SI()

    # The Sand Filter CV method takes place outside of the nManning's calibration for-loop
    # Qin_SFSH, Q_exfil = SandFilter_CV(y2_depth, y_time, Qex_flow, Qex_time)
    # Qin_GVPF = Gradually_varied_pipe_flow(y1_depth, y2_depth)


    Qin_energy = np.zeros(len(y_time))
    Qin_mannings = np.zeros(len(y_time))
    volumes = np.zeros(len(nMannings))
    for nM in range(len(nMannings)):
        for yy in range(len(y_time)):

            nMannings[nM] = 0.0185
            Qin_mannings[yy] = mannings_eq_flow(y1_depth[yy], y2_depth[yy], nMannings[nM])
            Qin_energy[yy] = Energy_eq_check(y1_depth[yy], y2_depth[yy], nMannings[nM], Qin_mannings[yy])

        volumes[nM] = volume_calc(Qin_energy, y_time)
        print("nMannings: ", str(round(nMannings[nM], 4)), "Inflow Volume", str(round(volumes[nM], 5)))

        # Qin_sample = match_sample_to_inflow(sample_time, y_time, Qin_energy)

        # emc_alloc = volume_alloc(sample_time, y_time, Qin_energy, composite_volume_total_L)
        # result = [round(item, 3) for item in emc_alloc]
        # print(result)

        fig, ax = plt.subplots()
        fig.suptitle('ASF In/Out Flowrates vs Time')
        # ax.plot(y_time, Qin_SFSH, color='Red', label='Qin_StorageCV')
        # ax.plot(y_time, Q_exfil, color='Green', label='Qout_Exfiltration')
        # ax.plot(Qex_time, Qex_flow, color='Green', label='Q_exfil')
        ax.plot(y_time, Qin_energy, color='Blue', label='Qin_EnergyEq')
        # ax.plot(sample_time, Qin_sample, 'o', color='Purple', label='Inflow Sampled')
        # ax.plot(y_time, Qin_mannings, color='Black', label='Qin_mannings')
        ax.set(xlabel="Time since start (s)", ylabel="Flowrate (cms)")
        ax.legend()
        plt.show()
        quit()
        # # fig.savefig("./Sensitivity Analysis for Manning's n/nMannings %s.png" % str(round(nMannings[nM], 4)))

    
    original_stdout = sys.stdout
    with open('Volumes_017_019.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        for nM in range(len(nMannings)):
            print("nMannings: ", str(round(nMannings[nM], 4)), "Inflow Volume", str(round(volumes[nM], 5)))
        sys.stdout = original_stdout # Reset the standard output to its original value
    return Qin_energy


# Qin_energy = determine_hydrographs(y1_depth, y2_depth, y_time, Qex_flow, Qex_time, nMannings, sample_time)
Qin_energy = determine_hydrographs(y1_depth, y2_depth, y_time, Qex_flow, Qex_time, nMannings, [])


