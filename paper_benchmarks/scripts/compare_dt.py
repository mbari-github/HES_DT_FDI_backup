#!/usr/bin/env python3
"""
Confronto di theta tra due simulazioni con dt differente.
Utilizzo:
  python3 compare_dt.py <bag_dt_1ms> <bag_dt_0.5ms> [--publish-dt 0.005] [--show-proj]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import rosbag2_py
from rclpy.serialization import deserialize_message
from exoskeleton_safety_msgs.msg import Float64ArrayStamped


def read_debug_topic(bag_path: str, storage_id='mcap'):
    """
    Legge /exo_dynamics/debug da una bag.
    Restituisce:
        time_sec: array tempi (s)
        theta: array (rad)
        proj:  array (N)   (campo 5 di debug: proj_last)
    """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    times = []
    theta_vals = []
    proj_vals = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/exo_dynamics/debug':
            msg = deserialize_message(data, Float64ArrayStamped)
            arr = np.array(msg.data, dtype=np.float64)
            # debug layout:
            # 0: theta, 1: theta_dot, 2: theta_ddot, 3: tau_m, 4: denom_last,
            # 5: proj_last, 6: gproj_last, 7: closure_norm, 8: success,
            # 9: nfev, 10: hit_bounds, 11: at_limit, 12: tau_ext_theta,
            # 13: tau_pass_theta, 14: reaction_theta_last
            theta_vals.append(arr[0])
            proj_vals.append(arr[5])
            times.append(timestamp * 1e-9)
    reader.close()

    return np.array(times), np.array(theta_vals), np.array(proj_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_1ms', help='Percorso bag con dt=1ms')
    parser.add_argument('bag_0_5ms', help='Percorso bag con dt=0.5ms')
    parser.add_argument('--publish-dt', type=float, default=0.005,
                        help='Intervallo di pubblicazione in s (default: 0.005)')
    parser.add_argument('--show-proj', action='store_true',
                        help='Confronta anche proj_last')
    args = parser.parse_args()

    print("Lettura bag dt=1ms ...")
    t1, th1, pr1 = read_debug_topic(args.bag_1ms)
    print("Lettura bag dt=0.5ms ...")
    t2, th2, pr2 = read_debug_topic(args.bag_0_5ms)

    # I due array potrebbero avere lunghezze leggermente diverse.
    # Allinea su un asse temporale comune: usiamo il tempo della bag a dt=1ms
    # e interpoliamo l'altra.
    th2_interp = interp1d(t2, th2, kind='linear', fill_value='extrapolate')(t1)
    pr2_interp = interp1d(t2, pr2, kind='linear', fill_value='extrapolate')(t1)

    # Errore
    err_th = th1 - th2_interp
    rmse_th = np.sqrt(np.mean(err_th**2))
    max_err = np.max(np.abs(err_th))
    print(f"\nConfronto θ: RMSE = {rmse_th:.6e} rad, |max error| = {max_err:.6e} rad")
    print(f"  Range θ: [{np.min(th1):.4f}, {np.max(th1):.4f}] rad")

    if args.show_proj:
        err_pr = pr1 - pr2_interp
        rmse_pr = np.sqrt(np.mean(err_pr**2))
        print(f"Confronto proj: RMSE = {rmse_pr:.6e} Nm")

    # Plot θ
    plt.figure(figsize=(10, 5))
    plt.plot(t1, th1, label='θ dt=1.0 ms', linewidth=1.5)
    plt.plot(t1, th2_interp, '--', label='θ dt=0.5 ms', linewidth=1.5)
    plt.xlabel('Tempo [s]')
    plt.ylabel('θ [rad]')
    plt.title(f'Confronto θ (publish ogni {args.publish_dt}s), RMSE = {rmse_th:.3e} rad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('theta_comparison.png', dpi=150)
    plt.show()

    # Plot errore
    plt.figure(figsize=(10, 3))
    plt.plot(t1, err_th, linewidth=0.8)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Δθ [rad]')
    plt.title('Differenza θ')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('theta_error.png', dpi=150)
    plt.show()

    if args.show_proj:
        plt.figure(figsize=(10, 4))
        plt.plot(t1, pr1, label='proj dt=1ms')
        plt.plot(t1, pr2_interp, '--', label='proj dt=0.5ms')
        plt.xlabel('Tempo [s]')
        plt.ylabel('proj [Nm]')
        plt.title('Confronto proj')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj_comparison.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    main()