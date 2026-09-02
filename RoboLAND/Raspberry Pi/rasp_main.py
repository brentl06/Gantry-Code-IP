import subprocess
import datetime

def launch_rviz(rviz_config):
    """Launches RViz with the given configuration.

    Args:
        rviz_config (str): Path to the RViz configuration file.

    Returns:
        subprocess.Popen: The RViz subprocess.
    """
    process = subprocess.Popen(["ros2", "run", "rviz2", "rviz2", "-d", rviz_config])
    return process


    
def launch_bridge():
    # Open /dev/null file
    with open('/dev/null', 'w') as dev_null:
        process = subprocess.Popen(["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"], stdout=dev_null, stderr=dev_null)
    return process

def launch_c_node(packagename, nodename):
    # print(packagename)
    # print(nodename)
    process = subprocess.Popen(["ros2", "run", packagename, nodename])
    return process


def launch_python_node(node_script_path):
    """Starts a node script as a separate process.

    Args:
        node_script_path (str): Path to the node script file.

    Returns:
        subprocess.Popen: The node script subprocess.
    """
    process = subprocess.Popen(['python3', node_script_path])
    return process


class NodeManager:
    def __init__(self):
        self.nodes = {}

    def register(self, name, path, is_bridge=False, isc=False):
        """Register a node."""
        if is_bridge:
            self.nodes[name] = launch_bridge()
        elif isc:
            self.nodes[name] = launch_c_node(name, path)
        else:
            self.nodes[name] = launch_python_node(path)

    def block(self):
        """Block main thread until all nodes are done."""
        try:
            [node.wait() for node in self.nodes.values()]
        except KeyboardInterrupt:
            [node.terminate() for node in self.nodes.values()]
            [node.wait() for node in self.nodes.values()]


def main():
    """Main function to launch RViz and start node scripts."""
    manager = NodeManager()
    # manager.register('bridge', None, is_bridge=True)
    # manager.register('state_estimator', './visualizer/stateEstimator.py')
    manager.register('traveler_high_controller', 'traveler_high_controller', isc=True)
    # manager.register('data_sync', './multimedia/dataSync.py')
    # manager.register('gui', './LASSIE_GUI/lassie_gui.py')
    manager.register('video_sync', '/home/traveler/highlevel/multimedia/videoSync.py')

    # print(manager.nodes)
    manager.block()



if __name__ == '__main__':
    main()
