"""Unfitted planar command-integration control for prediction comparisons.

This predicts ideal body-twist execution. It estimates neither the current
pose nor contact risk and does not model gait inertia or slip.
"""
import numpy as np
from lewm.terminal_translation_pulse_development import command_sequences


def forecast(prefix, *, pulse):
    prefix = np.asarray(prefix, float)
    if (prefix.shape != (3,3) or not np.isfinite(prefix).all() or
            np.any(prefix[:,1] != 0) or np.any(np.abs(prefix) > [.3,1.,.5]) or
            type(pulse) is not bool):
        raise ValueError('three bounded committed commands and explicit pulse mode required')
    commands = command_sequences(prefix, pulse=pulse)
    result = np.zeros((6,8,4), dtype=float)
    xy = np.zeros((6,2)); yaw = np.zeros(6)
    for tick in range(8):
        velocity = commands[:,tick,:2]; angle = commands[:,tick,2]*.1
        sine_ratio = np.sinc(angle/np.pi)
        cosine_ratio = .5*angle*np.sinc(angle/(2*np.pi))**2
        local = .1*np.column_stack((sine_ratio*velocity[:,0]-cosine_ratio*velocity[:,1],
            cosine_ratio*velocity[:,0]+sine_ratio*velocity[:,1]))
        c,s = np.cos(yaw),np.sin(yaw)
        xy += np.column_stack((c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1]))
        yaw += angle
        result[:,tick,:2] = xy
        result[:,tick,2] = np.sin(yaw);result[:,tick,3] = np.cos(yaw)
    return result
