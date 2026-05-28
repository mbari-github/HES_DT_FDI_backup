#!/usr/bin/env python3
"""
Plotta le componenti di B e Bdot da una rosbag2.
Utilizzo: python3 plot_B_Bdot.py <percorso_bag> [--publish-dt DT] [--storage-id STORAGE]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from exoskeleton_safety_msgs.msg import Float64ArrayStamped


def read_bag(bag_path: str, storage_id: str = 'mcap'):
    """
    Legge /exo_dynamics/B e /exo_dynamics/Bdot dalla bag.
    Restituisce:
        timestamps_sec: array dei tempi in secondi
        B_all: array (Nsamples, nv) delle componenti di B
        Bdot_all: array (Nsamples, nv) delle componenti di Bdot


    USAGE:
        mbari@myPC:~/ros2_ws$ python3 src/HES_DT_FDI/paper_benchmarks/scripts/plot_B_Bdot.py /home/mbari/b_and_bdot_bag

        """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    B_list = []
    Bdot_list = []
    time_list = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/exo_dynamics/B':
            msg = deserialize_message(data, Float64ArrayStamped)
            B_list.append(np.array(msg.data, dtype=np.float64))
            time_list.append(timestamp * 1e-9)  # da nanosecondi a secondi
        elif topic == '/exo_dynamics/Bdot':
            msg = deserialize_message(data, Float64ArrayStamped)
            Bdot_list.append(np.array(msg.data, dtype=np.float64))

    reader.close()

    # Allinea e taglia alla lunghezza minima (i due topic dovrebbero avere lo stesso numero di messaggi)
    min_len = min(len(B_list), len(Bdot_list))
    timestamps_sec = np.array(time_list[:min_len])
    B_all = np.array(B_list[:min_len])
    Bdot_all = np.array(Bdot_list[:min_len])

    return timestamps_sec, B_all, Bdot_all


def plot_signals(timestamps_sec, B, Bdot, publish_dt, use_time=False):
    """
    Crea due subplot: componenti di B e Bdot in funzione del passo k o del tempo.
    """
    nv = B.shape[1]
    if use_time:
        x = timestamps_sec
        xlabel = 'Tempo [s]'
    else:
        k = np.arange(len(timestamps_sec))
        x = k
        xlabel = 'Passo k (ogni {:.3f} s)'.format(publish_dt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for j in range(nv):
        ax1.plot(x, B[:, j], label=f'B[{j}]', linewidth=0.8)
    ax1.set_ylabel('Componenti di B')
    ax1.set_title('Vettore di proiezione B (publish_dt = {:.3f} s)'.format(publish_dt))
    ax1.legend(loc='best', ncol=2, fontsize='small')
    ax1.grid(True, alpha=0.3)

    for j in range(nv):
        ax2.plot(x, Bdot[:, j], label=f'Bdot[{j}]', linewidth=0.8)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Componenti di Bdot')
    ax2.set_title('Derivata temporale Bdot')
    ax2.legend(loc='best', ncol=2, fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plotta B e Bdot da rosbag2')
    parser.add_argument('bag_path', type=str,
                        help='Percorso assoluto della directory della bag')
    parser.add_argument('--publish-dt', type=float, default=0.005,
                        help='Intervallo di pubblicazione in secondi (default: 0.005)')
    parser.add_argument('--use-time', action='store_true',
                        help='Usa il tempo (secondi) anziché i passi k')
    parser.add_argument('--storage-id', type=str, default='mcap',
                        help="Formato della bag ('mcap' o 'sqlite3', default: 'mcap')")
    args = parser.parse_args()

    print(f"Lettura della bag: {args.bag_path} (formato: {args.storage_id})")
    timestamps, B, Bdot = read_bag(args.bag_path, args.storage_id)
    print(f"Numero campioni: {len(timestamps)}")
    print(f"Numero componenti vettore B: {B.shape[1]}")
    print(f"Intervallo temporale: {timestamps[0]:.2f} - {timestamps[-1]:.2f} s")

    plot_signals(timestamps, B, Bdot, args.publish_dt, use_time=args.use_time)


if __name__ == '__main__':
    main()