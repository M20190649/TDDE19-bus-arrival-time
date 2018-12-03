import numpy as np

def kalman_filter(observations, sigma_0=100, R=2, Q=5):
    n = len(observations)

    mus    = np.zeros(n)
    sigmas = np.zeros(n)
    errors = np.zeros(n)

    mus[0] = observations[0]
    sigmas[0] = sigma_0

    for t in range(1,n):
        # Prediction
        mu_bar    = mus[t-1] - 1 # -1 since tta should decrease every seconds?
        sigma_bar = sigmas[t-1] + np.power(R, 2)

        # Kalman gain
        K = sigma_bar / (sigma_bar + np.power(Q, 2))

        # Correction
        mus[t] = mu_bar + K * (observations[t] - mu_bar)
        sigmas[t] = (1 - K) * sigma_bar

        errors[t] = abs(observations[t] - mus[t])

    #print("Mean of prediction deviation:", np.mean(errors), "s")
    return mus, errors