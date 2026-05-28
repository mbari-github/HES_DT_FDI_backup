#!/usr/bin/env python3
"""
Nodo per input di forza esterna con profilo trapezoidale.
Sostituisce la GUI a slider per test riproducibili (es. confronto a diversi dt).

Parametri ROS2:
  delay       (s)   – ritardo prima dell'inizio della rampa (default 1.0)
  rise_time   (s)   – durata della rampa di salita (default 0.5)
  hold_time   (s)   – durata del tratto costante (default 1.0)
  fall_time   (s)   – durata della rampa di discesa (default 0.5)
  max_force   (N)   – valore massimo della forza (default 20.0)
  publish_rate (Hz) – frequenza di pubblicazione (default 100)

Il profilo rimane a zero dopo la discesa.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped


class TrapezoidForce(Node):
    def __init__(self):
        super().__init__('trapezoid_force')

        # Parametri del profilo
        self.declare_parameter('delay', 4.0)
        self.declare_parameter('rise_time', 2.0)
        self.declare_parameter('hold_time', 8.0)
        self.declare_parameter('fall_time', 2.0)
        self.declare_parameter('max_force', -10.0)
        self.declare_parameter('publish_rate', 100.0)

        self.delay = self.get_parameter('delay').value
        self.rise = self.get_parameter('rise_time').value
        self.hold = self.get_parameter('hold_time').value
        self.fall = self.get_parameter('fall_time').value
        self.max_f = self.get_parameter('max_force').value
        self.rate = self.get_parameter('publish_rate').value

        # Calcolo tempi di fine fase
        self.t1 = self.delay                     # fine ritardo
        self.t2 = self.t1 + self.rise            # fine salita
        self.t3 = self.t2 + self.hold            # fine mantenimento
        self.t4 = self.t3 + self.fall            # fine discesa

        self.publisher = self.create_publisher(WrenchStamped, '/exo_dynamics/external_wrench', 10)
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

        self.start_time = self.get_clock().now()
        self.get_logger().info(
            f"Trapezoid force node started: "
            f"delay={self.delay}s, rise={self.rise}s, hold={self.hold}s, fall={self.fall}s, "
            f"max={self.max_f}N"
        )

    def compute_force(self, elapsed: float) -> float:
        """Restituisce il valore di forza in base al tempo trascorso in secondi."""
        if elapsed < self.t1:
            return 0.0
        elif elapsed < self.t2:
            # rampa di salita lineare
            return self.max_f * (elapsed - self.t1) / self.rise
        elif elapsed < self.t3:
            return self.max_f
        elif elapsed < self.t4:
            # rampa di discesa lineare
            return self.max_f * (1.0 - (elapsed - self.t3) / self.fall)
        else:
            return 0.0

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9
        force_z = self.compute_force(elapsed)

        msg = WrenchStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'world'
        msg.wrench.force.x = 0.0
        msg.wrench.force.y = 0.0
        msg.wrench.force.z = force_z
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = 0.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidForce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()