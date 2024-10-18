import numpy as np
import matplotlib.pyplot as plt

def plot_flowpic(flowpic):
    # Extract the histogram component from the FlowPic
    histogram = flowpic[:, :, 0]

    # Plot the histogram using imshow
    plt.imshow(histogram, cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Packet Size')
    plt.ylabel('Arrival Time')
    plt.title('FlowPic 2D Histogram')
    plt.colorbar(label='Normalized Frequency')
    plt.show()

# Generate a sample FlowPic
# Replace df with your DataFrame
# flowpic_data = flowpic(df)
# plot_flowpic(flowpic_data)

def flowpic(df, bins=1500):
    """
    Takes in a DataFrame of per-packet measurements and creates a FlowPic. A
    FlowPic is essentially a 2D Histogram of packet size (up to 1500 Bytes) and
    arrival time (normalized from 0 to 1500). The FlowPic also has an additional
    dimension for the proportion of packets that were downloaded in each bin.

    NOTE: Currently the direction channel is filled with 0.5... not sure how
    fine of an idea this is honestly.

    """

    # Ignoring packet_dirs currently
    #c = df[['packet_dirs', 'packet_times', 'packet_sizes']]
    c = df[['Timestamp', 'Packet size']]

    c = c[c['Packet size'] <= 1500]
    c.Timestamp = c.Timestamp - c.Timestamp.min()
    c.Timestamp = c.Timestamp / c.Timestamp.max() * 1500

    hist_bins=bins
    binrange = [[0, 1500], [0, 1500]]
    hist = np.histogram2d(c.Timestamp, c['Packet size'], bins=hist_bins, range=binrange)
    h = hist[0]
    if h.max() > 0:
        h = h / h.max()
    '''
    else:    
        print(c.Timestamp)
        print(c['Packet size'])
        print("No data in bins, normalization skipped.")
        raise ValueError("Division by zero is not allowed")
    '''
    #h = h / h.max()

    # For each bin we
    # want to calculate the proportion of packets that were down-
    # loaded. A 1.0 means all packets in that bin were downloaded packets, and 0.0
    # means all packets in that bin were uploaded.
    cut_bins = np.arange(bins)
    timebins = np.searchsorted(cut_bins, c.Timestamp, side='right') - 1
    sizebins = np.digitize(c['Packet size'], cut_bins, right=False) - 1
    c['bin'] = list(zip(timebins, sizebins))
    '''
    c.packet_dirs = c.packet_dirs - 1
    download_props = c.groupby('bin').packet_dirs.mean()
    prop_bins = download_props.index.values
    '''

    # Start off with a 'grey' channel -- all values are equally up- and downloaded.
    #download_channel = np.full((1500,1500), 0.5)

    # Then fill in the calculated values from above
    #download_channel[list(zip(*prop_bins))] = download_props.values

    #dc = download_channel

    #flowpic = np.dstack([h, dc])

    flowpic = h

    #plot_flowpic(flowpic)

    return flowpic