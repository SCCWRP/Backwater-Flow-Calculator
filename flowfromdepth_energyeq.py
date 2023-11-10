# Dependencies
import numpy as np
import pandas as pd
import sys
import scipy
from matplotlib import pyplot as plt
from bisect import bisect_left

# Hard-coded project parameters - Need to read from template
radius_in = 6
length_ft = 27.583
nMannings = 0.0185
nMannings_arr = np.arange(0.015, 0.4, 0.0005)
slope = 0.002
composite_volume_total_L = 3
porosity = 0.3
surface_area = 1262
depth_of_sand = 1.5

# Hard-coded program parameters, including conversion factors
gravity_SI = 9.81
sf_to_m2 = 0.0929
in_to_m = 0.0254
ft_to_m = 0.3048
min_to_sec = 60.0
cfs_to_cms = 0.0283
Q_guess_factor = 2
Q_guess_interval = 1000
depth_uncertainty_in = 0.0787


sheet_name_var = 'Mar 9 - 17_FULL'

# Use pandas to read data template
data = pd.read_excel(r'C:/Users/edwardt/SCCWRP/SMC BMP Projects - Regional monitoring network/Products - Calculators/Two-Depth Flow Calculator/Flow_calculator_repo/FlowCalc_Template.xlsx', \
    sheet_name=sheet_name_var)
df_depth = pd.DataFrame(data, columns=['Elapsed Time Depth (min)', 'y1 (in)', 'y2 (in)'])
df_flow = pd.DataFrame(data, columns=['Elapsed Time Flow (min)', 'Exfiltration (cfs)'])
df_sample = pd.DataFrame(data, columns=['Elapsed Time Sample (min)'])

# Use pandas to print data to template
output_file = "Caltrans 1194 - Dec 12.csv"
sheet_hydrograph = "Hydrograph"
sheet_EMC = "EMC composite"

# Strip out NA's from dataframes, store as 1D-list variables
y1_depth_raw = df_depth['y1 (in)'][df_depth['y1 (in)'].notna()]
y2_depth_raw = df_depth['y2 (in)'][df_depth['y2 (in)'].notna()]
y_time = df_depth['Elapsed Time Depth (min)'][df_depth['Elapsed Time Depth (min)'].notna()]
Qex_time = df_flow['Elapsed Time Flow (min)'][df_flow['Elapsed Time Flow (min)'].notna()]
Qex_flow = df_flow['Exfiltration (cfs)'][df_flow['Exfiltration (cfs)'].notna()]

sample_time = df_sample['Elapsed Time Sample (min)'][df_sample['Elapsed Time Sample (min)'].notna()]

y1_depth = scipy.signal.medfilt(y1_depth_raw, 3)
y2_depth = scipy.signal.medfilt(y2_depth_raw, 3)

y1_surge = np.zeros(len(y1_depth))
y1_back = np.zeros(len(y1_depth))
y2_surge = np.zeros(len(y2_depth))
y2_back = np.zeros(len(y2_depth))

def settings_SI():
    global radius_m, length_m, zz_drop_m, area_full_m2, wp_full_m, rh_full_m

    # Convert units to SI
    radius_m = radius_in * in_to_m
    length_m = length_ft * ft_to_m

    zz_drop_m = slope * length_m
    area_full_m2 = np.pi * radius_m**2
    wp_full_m = np.pi * radius_m * 2
    rh_full_m = area_full_m2/wp_full_m

    # Convert min -> sec, in -> m, cfs -> cms
    for ii in range(len(y_time)):
        y_time[ii] = y_time[ii] * min_to_sec
        y1_depth[ii] = y1_depth[ii] * in_to_m
        y2_depth[ii] = y2_depth[ii] * in_to_m

        # To evaluate range of uncertainty, extremes of instrument precision are investigated
        # _surge indicates flow conditions that would promote flows (i.e. delta y large)
        # _back indicates flow conditions that retard flows (i.e. delta y small)
        y1_surge[ii] = y1_depth[ii] + depth_uncertainty_in*in_to_m
        y2_surge[ii] = y2_depth[ii] - depth_uncertainty_in*in_to_m
        y1_back[ii] = y1_depth[ii] - depth_uncertainty_in*in_to_m
        y2_back[ii] = y2_depth[ii] + depth_uncertainty_in*in_to_m

        # If instrument-recorded depths are negative, set them to zero
        if y1_depth[ii] < 0:
            print("Zero Depth found and corrected, y1")
            y1_depth[ii] = 0
        if y2_depth[ii] < 0:
            print("Zero Depth found and corrected, y2")
            y2_depth[ii] = 0
        if y1_surge[ii] < 0:
            y1_surge[ii] = 0
        if y1_back[ii] < 0:
            y1_back[ii] = 0
        if y2_surge[ii] < 0:
            y2_surge[ii] = 0
        if y2_back[ii] < 0:
            y2_back[ii] = 0

    # Convert min -> sec, cfs -> cms
    for ii in range(len(Qex_time)):
        Qex_time[ii] = float(Qex_time[ii] * min_to_sec)
        Qex_flow[ii] = Qex_flow[ii] * cfs_to_cms
    
    for ii in range(len(sample_time)):
        sample_time[ii] = sample_time[ii] * min_to_sec

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
    area_ave = flow_area(y_ave, theta_ave)

    # Calculate the wetted perimeter from flow depth and internal angle
    wp_ave = wetted_perimeter(y_ave, theta_ave)

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
    # Q_low = 0
    Q_high = max(Q_mannings_guess * Q_guess_factor, 1e-9)
    Q_interval = (Q_high - Q_low) / Q_guess_interval
    potential_flowrates = np.arange(Q_low, Q_high, Q_interval)

    if Q_low < 0:
        print("Q_low too low: ", Q_low)
        quit()


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
    # print(area_ave, rh_ave)
    
    """The energy equation is composed of 4 head terms: Velocity1, Velocity2, Friction Loss, Elevation
        of which 3 are functions of flow rate: Velocity1, Velocity2, Friction Loss

    Velocity1(Q) - Velocity2(Q) - Friction Loss(Q) + Elevation = residual

    ^The flowrate that minimizes the residual is considered the instantaneous flow rate
    """
    # Elevation head is the free surface elevation difference between y2 and y1 (from raw data, does not depend on Q)
    elev_head_diff = y1 - y2 + zz_drop_m

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
        new_energy_residual = np.abs(vel_head1 - vel_head2 - fric_head + elev_head_diff)
        if new_energy_residual < old_energy_residual:
            old_energy_residual = new_energy_residual
            qq_target = qq
    
    # print(qq_target, nM, vel_head1, vel_head2, fric_head, elev_head_diff, new_energy_residual)
    # quit()

    # Flowrate Q such that residual is minimized
    return qq_target


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
        # theta = np.arccos(1 - reference_depth/radius_m)
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
    small_depth_threshold = radius_m * 0.0001
    if 0 <= depth < small_depth_threshold:
        area = radius_m * depth

    # Else-if flow depth is less than the radius, calculate area directly
    elif small_depth_threshold <= depth < radius_m:
        area = radius_m**2*(theta - np.sin(theta))/2

    # Else-if flow depth is less than the diameter, calculate area as full - empty area
    elif radius_m <= depth < diamter_m:
        area = area_full_m2 - radius_m**2*(theta - np.sin(theta))/2

    # Else-if flow depth is greater than the diameter, the area is full and the full_area_flag is turned on
    elif depth >= diamter_m:
        area = area_full_m2
        full_area_flag = True
        print("Full area")
    
    # Error condition, stop
    else:
        area = 0
        print("Check the depth condition, something has gone wrong")
        print(depth)
        quit()

    area = max(area, 1E-10)
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
        wp = radius_m * theta

    # If more than half full
    elif radius_m <= depth < diamter_m:
        wp = full_perimeter - radius_m * theta

    # If more than completely full
    elif depth >= diamter_m:
        wp = full_perimeter

    # Anything else means something is wrong with the depth condition
    else:
        wp = 0
        "Check the depth condition, something has gone wrong"
        print(depth)
        quit()

    wp = max(wp, 1E-10)
    return wp


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
            print(Qex_time[ii], Qex_flow[ii])
    # for ii in range(len(y_time)):
    #     ytime = y_time[ii]
    #     print(ytime)
    #     index = Qex_time.index(take_closest(y_time, Qex_time[ii]))
    #     print(Qex_time[index])
    #     Q_exfil_DepthMask.append(Qex_flow[index])

    return Q_exfil_DepthMask


def volume_calc(y_series, x_series):
    """
    This subroutine calculates the trapezoidal approximation of the area under the curve
    """

    volume = np.trapz(y_series, x_series)
    return volume


def match_sample_to_inflow(sample_timeseries, y_time, Qinflow):
    """
    This subroutine is used to allocate the inflow volume to individual samples in the sample timeseries

    Each sample is allocated the portion of the total inflow hydrograph that is nearest in time to that sample (Central Attribution)
    """
    Q_sample = []
    sample_timeseries = list(sample_timeseries)
    y_time = list(y_time)
    Qinflow = list(Qinflow)

    # When the outflow timeseries elapsed time is aligned with the depth timeseries elapsed time, include value in the masked outflow timeseries
    for ii in range(len(sample_timeseries)):
        index = y_time.index(take_closest(y_time, sample_timeseries[ii]))
        Q_sample.append(Qinflow[index])

    return Q_sample


def volume_alloc(sample_timeseries, y_time, Qinflow, Comp_Vol):
    """
    This subroutine calculates the volume of hydrograph that was associated with each sample.

    The result is a list of volumes (len(sample_timeseries)) and a list of proportions that sum to 1
    """

    # Initialize sample timeseries
    sample_timeseries = list(sample_timeseries)
    sample_volumes = []
    y_time = list(y_time)
    Qinflow = list(Qinflow)
    sample_timeseries = [0] + sample_timeseries + [y_time[-1]]
    first_gap = sample_timeseries[1] - sample_timeseries[0]
    last_gap = sample_timeseries[-1]

    # If the first sample is not at the start of the inflow hydrograph, calculate the volume associated with the gap
    sample_leftedge = 0
    time_index_left = 0
    sample_rightedge = np.average([sample_timeseries[1], sample_timeseries[2]])
    time_index_right = y_time.index(take_closest(y_time, sample_rightedge))

    Qinflow_snip = Qinflow[time_index_left:time_index_right+1]
    y_time_snip = y_time[time_index_left:time_index_right+1]
    volume_snip = volume_calc(Qinflow_snip, y_time_snip)
    sample_volumes.append(volume_snip)

    # Calculate the volume associated with each sample
    for ii in range(2, len(sample_timeseries)-2):

        sample_leftedge = np.average([sample_timeseries[ii-1], sample_timeseries[ii]])
        time_index_left = y_time.index(take_closest(y_time, sample_leftedge))

        sample_rightedge = np.average([sample_timeseries[ii], sample_timeseries[ii+1]])
        time_index_right = y_time.index(take_closest(y_time, sample_rightedge))
        
        Qinflow_snip = Qinflow[time_index_left:time_index_right+1]
        y_time_snip = y_time[time_index_left:time_index_right+1]
        volume_snip = volume_calc(Qinflow_snip, y_time_snip)
        sample_volumes.append(volume_snip)

    # If the last sample is not at the end of the inflow hydrograph, calculate the volume associated with the gap
    sample_leftedge = np.average([sample_timeseries[-3], sample_timeseries[-2]])
    time_index_left = y_time.index(take_closest(y_time, sample_leftedge))

    Qinflow_snip = Qinflow[time_index_left:]
    y_time_snip = y_time[time_index_left:]
    volume_snip = volume_calc(Qinflow_snip, y_time_snip)
    sample_volumes.append(volume_snip)

    vol_total = sum(sample_volumes)
    vol_proportion = sample_volumes/vol_total
    volume_allocated = np.empty(len(sample_volumes))
    for ss in range(len(sample_volumes)):
        volume_allocated[ss] = Comp_Vol * sample_volumes[ss] / vol_total
    return volume_allocated, vol_proportion


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

    for ii in range(len(y1_surge)):
        if y1_surge[ii] < 0:
            print("y1_surge", y1_surge[ii])
        if y1_surge[ii] < 0:
            print("y1_back", y1_back[ii])
        if y2_back[ii] < 0:
            print("y2_back", y2_back[ii])
        if y2_surge[ii] < 0:
            print("y2_surge", y2_surge[ii])


    # # To evaluate range of uncertainty, extremes of instrument precision are investigated
    # # _surge indicates flow conditions that would promote flows (i.e. delta y large)
    # # _back indicates flow conditions that retard flows (i.e. delta y small)
    # y1_surge = y1_depth + depth_uncertainty_in*in_to_m
    # y2_surge = y2_depth - depth_uncertainty_in*in_to_m
    # y1_back = y1_depth - depth_uncertainty_in*in_to_m
    # y2_back = y2_depth + depth_uncertainty_in*in_to_m

    # Q_exfil = timeseries_mask_to_depth(y_time, Qex_flow, Qex_time)

    Qin_energy = np.zeros(len(y_time))
    Qin_mannings = np.zeros(len(y_time))
    Qin_mannings_low = np.zeros(len(y_time))
    Qin_mannings_high = np.zeros(len(y_time))
    Qin_energy_surge = np.zeros(len(y_time))
    Qin_energy_back = np.zeros(len(y_time))
    theta_arr = np.zeros(len(y_time))
    area_arr = np.zeros(len(y_time))
    wp_arr = np.zeros(len(y_time))
    rh_arr = np.zeros(len(y_time))
    elev_arr = np.zeros(len(y_time))
    
    for yy in range(len(y_time)):

        # Pipe-flow geometry parameters calculated here
        theta_arr[yy] = theta_calc(y1_depth[yy])
        area_arr[yy] = flow_area(y1_depth[yy], theta_arr[yy])
        wp_arr[yy] = wetted_perimeter(y1_depth[yy], theta_arr[yy])
        elev_arr[yy] = y1_depth[yy] - y2_depth[yy] + zz_drop_m

    # print(radius_m)
    # # Scatter plot of theta_arr vs elev_arr
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_time, theta_arr, color='blue')

    # # Time series for area_arr and wp_arr
    # plt.plot(y_time, area_arr, color='red', label='Area')
    # plt.plot(y_time, wp_arr, color='green', label='Wetted Perimeter')

    # plt.xlabel('y1 depth')
    # plt.ylabel('Theta')
    # # plt.title('Elevation vs Time')
    # plt.show()
    # quit()

        Qin_mannings[yy] = mannings_eq_flow(y1_depth[yy], y2_depth[yy], nMannings)
        Qin_mannings_low[yy] = mannings_eq_flow(y1_depth[yy], y2_depth[yy], 0.007)
        Qin_mannings_high[yy] = mannings_eq_flow(y1_depth[yy], y2_depth[yy], 0.025)

        Qin_energy[yy] = Energy_eq_check(y1_depth[yy], y2_depth[yy], nMannings, Qin_mannings[yy])
        Qin_energy_surge[yy] = Energy_eq_check(y1_surge[yy], y2_surge[yy], nMannings, Qin_mannings[yy])
        Qin_energy_back[yy] = Energy_eq_check(y1_back[yy], y2_back[yy], nMannings, Qin_mannings[yy])

        # if (y_time[yy]/60) > 3500:
        #     print(y_time[yy]/60, y1_depth[yy]/in_to_m, y2_depth[yy]/in_to_m, Qin_mannings[yy], Q_exfil[yy])

        # if (y_time[yy]/60) > 4000:
        #     quit()

    Qin_sample = match_sample_to_inflow(sample_time, y_time, Qin_energy)

    v_retained = porosity * (surface_area * depth_of_sand) * ft_to_m**3
    print("Calc Vol Out", volume_calc(Qex_flow, Qex_time)/cfs_to_cms)
    print("Est Vol Retained", v_retained/cfs_to_cms)
    print("Cal Vol In", volume_calc(Qin_mannings, y_time)/cfs_to_cms)
    print("Cal Vol In Low", volume_calc(Qin_mannings_low, y_time)/cfs_to_cms)
    print("Cal Vol In High", volume_calc(Qin_mannings_high, y_time)/cfs_to_cms)

    # emc_alloc, vol_proportion = volume_alloc(sample_time, y_time, Qin_energy, composite_volume_total_L)
    # result_vol = [1000*round(item, 3) for item in emc_alloc]
    # result_alloc = [round(item, 3) for item in vol_proportion]
    # print(result_vol)
    # print(nMannings)
    # print(Qin_energy/cfs_to_cms)

    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle('ASF Influent/Effluent Flowrates vs Time')
    ax.plot(y_time/min_to_sec, Qin_energy, 'b-', label='Qin_EnergyEq')
    # ax.plot(y_time/min_to_sec, Qin_mannings, 'k-', label='Qin, Mannings n = ' + str(nMannings))
    # ax.plot(y_time/min_to_sec, Qin_mannings_low, 'b-', label='Qin, Mannings n = 0.007')
    # ax.plot(y_time/min_to_sec, Qin_mannings_high, 'r-', label='Qin, Mannings n = 0.025')
    ax.plot(y_time/min_to_sec, Qin_energy_back, 'g-', label='Qin_Back')
    ax.plot(y_time/min_to_sec, Qin_energy_surge, 'r-', label='Qin_Surge')
    # ax.plot(sample_time/min_to_sec, Qin_sample, 'o', color='Purple', label='Inflow Sampled')
    ax.plot(Qex_time/min_to_sec, Qex_flow, 'g--', label='Qout')
    # ax.plot(y_time/min_to_sec, Qin_mannings, color='Black', label='Qin_mannings')
    ax.set(xlabel="Time since start (min)", ylabel="Flowrate (cms)")
    ax.legend(loc='upper right')

    # ax2 = ax.twinx()
    # ax2.set_ylim(0, 12.0)
    # ax2.set_ylabel("Depth (in)")
    # ax2.plot(y_time/min_to_sec, y1_depth/in_to_m + zz_drop_m/in_to_m, 'k+', label='y1 elevation')
    # ax2.plot(y_time/min_to_sec, y2_depth/in_to_m, 'r+', label='y2 elevation')
    # ax2.legend(loc='upper right')
    
    plt.show()
    # # fig.savefig("./Sensitivity Analysis for Manning's n/nMannings %s.png" % str(round(nMannings[nM], 4)))

    print(len(sample_time))

    # Create a DataFrame
    df = pd.DataFrame({
        # 'y_time': y_time/min_to_sec,
        # 'Qin_energy': Qin_energy/cfs_to_cms,
        'Qex_time': Qex_time/min_to_sec,
        'Qex_flow': Qex_flow/cfs_to_cms
    })

    # Save the DataFrame to a CSV file
    df.to_csv('output.csv', index=False)

    # df = pd.DataFrame(result_alloc)
    # df.to_csv("Test4.csv")
    # df_1 = pd.DataFrame({"Inflow": Qin_energy, "Time": y_time}, columns=["Inflow (cms)", "Elapsed Time (min)"])
    # df_2 = pd.DataFrame({"Sample Time (min)": sample_time, "EMC (mL)": result_vol}, columns=["Inflow (cms)", "Elapsed Time (min)"])
    # df_1.to_csv(output_file)
    # df_2.to_csv(output_file, sheet_EMC)

    # original_stdout = sys.stdout
    # with open('Volumes_017_019.txt', 'w') as f:
    #     sys.stdout = f # Change the standard output to the file we created.
    #     for nM in range(len(nMannings)):
    #         print("nMannings: ", str(round(nMannings[nM], 4)), "Inflow Volume", str(round(volumes[nM], 5)))
    #     sys.stdout = original_stdout # Reset the standard output to its original value
    return Qin_energy


def calibrate_n(y1_depth, y2_depth, y_time, nMannings_arr):
    settings_SI()

    # Validated against Excel trapezoidal integration
    v_out = volume_calc(Qex_flow, Qex_time)
    print("Calculated Volume Out:", v_out)

    v_retained = porosity * (surface_area * depth_of_sand) * ft_to_m**3
    print("Estimated Volume Retained:", v_retained)

    Qin_energy = np.zeros(len(y_time))
    Qin_mannings = np.zeros(len(y_time))
    volumes = np.zeros(len(nMannings_arr))

    vol_diff_min = np.inf
    nMannings_calibrated = '0'

    # Iterate through the range of Mannning's n values, calculate the volume difference between inflow and outflow
    for nM in range(len(nMannings_arr)):
        
        # For each depth pair, calculate the inflow from Manning's equation and Energy equation
        for yy in range(len(y_time)):

            # Manning's equation applies to average depth
            Qin_mannings[yy] = mannings_eq_flow(y1_depth[yy], y2_depth[yy], nMannings_arr[nM])

            # Use Manning's equation result as a seed for the Energy equation
            Qin_energy[yy] = Energy_eq_check(y1_depth[yy], y2_depth[yy], nMannings_arr[nM], Qin_mannings[yy])

        volumes[nM] = volume_calc(Qin_energy, y_time)

        vol_diff = np.absolute(volumes[nM] - v_out - v_retained)
        if vol_diff < vol_diff_min:
            vol_diff_min = vol_diff
        elif vol_diff > vol_diff_min:
            print("nMannings: ", nMannings_calibrated, "Inflow Volume", str(round(volumes[nM], 5)))
            break
        
        nMannings_calibrated = str(round(nMannings_arr[nM], 4))
        print("nMannings: ", nMannings_calibrated, "Inflow Volume", str(round(volumes[nM], 5)))

    print("Calc Vol Out", volume_calc(Qex_flow, Qex_time)/cfs_to_cms)
    print("Est Vol Retained", v_retained/cfs_to_cms)
    print("Cal Vol In", volume_calc(Qin_energy, y_time)/cfs_to_cms)

    # Qin_sample = match_sample_to_inflow(sample_time, y_time, Qin_energy)

        # emc_alloc = volume_alloc(sample_time, y_time, Qin_energy, composite_volume_total_L)
        # result = [round(item, 3) for item in emc_alloc]
        # print(result)

    fig, ax = plt.subplots()
    fig.suptitle('ASF In/Out Flowrates vs Time - Nov 7-10 2022 \n' + 'Calibrated Mannings n = ' + nMannings_calibrated)
    # ax.plot(y_time, Qin_SFSH, color='Red', label='Qin_StorageCV')
    ax.plot(Qex_time/min_to_sec, Qex_flow, color='Green', label='Qout_Exfiltration')
    # ax.plot(Qex_time/min, Qex_flow, color='Green', label='Q_exfil')
    ax.plot(y_time/min_to_sec, Qin_energy, color='Blue', label='Qin_EnergyEq')
    # ax.plot(sample_time/min_to_sec, Qin_sample, 'o', color='Purple', label='Inflow Sampled')
    ax.plot(y_time/min_to_sec, Qin_mannings, color='Black', label='Qin_mannings')
    ax.set(xlabel="Time since start (min)", ylabel="Flowrate (cms)")
    ax.legend()
    plt.show()
    # quit()
        # # fig.savefig("./Sensitivity Analysis for Manning's n/nMannings %s.png" % str(round(nMannings[nM], 4)))

    # DF = pd.DataFrame(Qin_energy/cfs_to_cms)
    # DF.to_csv(sheet_name_var + ".csv")
    # original_stdout = sys.stdout
    # with open('Volumes_009_033.txt', 'w') as f:
    #     sys.stdout = f # Change the standard output to the file we created.
    #     for nM in range(len(nMannings_arr)):
    #         print("nMannings: ", str(round(nMannings_arr[nM], 4)), "Inflow Volume", str(round(volumes[nM], 5)))
    #     sys.stdout = original_stdout # Reset the standard output to its original value
    return nMannings_calibrated


# nMannings = calibrate_n(y1_depth, y2_depth, y_time, nMannings_arr)
nMannings = 0.0225
Qin_energy = determine_hydrographs(y1_depth, y2_depth, y_time, Qex_flow, Qex_time, nMannings, sample_time)


