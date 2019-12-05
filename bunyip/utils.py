    # def get_single_phase(self, t, y, yerr=None):
        
    #     if yerr is None:
    #         yerr = np.zeros_like(t)

    #     m = np.zeros(len(t), dtype=bool)
    #     period_grid = np.exp(np.linspace(np.log(2), np.log(50), 50000))
    #     bls_results = []
    #     periods = []
    #     t0s = []
    #     depths = []
    #     durations = []
    #     stats = []
    #     # Compute the periodogram for each star by iteratively masking out
    #     # eclipses from the higher signal to noise star.
    #     for i in range(2):
    #         bls = BoxLeastSquares(t[~m], y[~m])
    #         bls_power = bls.power(period_grid, 0.1, oversample=20)
    #         bls_results.append(bls_power)
            
    #         index = np.argmax(bls_power.power)
    #         periods.append(bls_power.period[index])
    #         t0s.append(bls_power.transit_time[index])
    #         depths.append(bls_power.depth[index])
    #         durations.append(bls_power.duration[index])

    #         stats.append(bls.compute_stats(periods[i], durations[i], t0s[i]))

    #         # Mask the data points that are in transit for this candidate
    #         m |= bls.transit_mask(t, periods[-1], 0.5, t0s[-1])
    #     mask = (t > (t[0] + t0s[0]-0.5*periods[0])) & (t < (t[0] + t0s[0]+0.5*periods[0]))
    #     t_transit = t[mask][::1]
    #     y_transit = y[mask][::1]
    #     yerr_transit = yerr[mask][::1]
        
    #     return t_transit, y_transit, yerr_transit


    # def detrend

    # def fold(self, period=None):
    #     pass
    
    # def polyfit(self, bins=101, method='mean'):
    #     pass