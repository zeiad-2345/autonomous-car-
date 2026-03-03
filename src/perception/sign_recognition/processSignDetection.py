if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.perception.sign_recognition.threads.threadSignDetection import threadSignDetection


class processSignDetection(WorkerProcess):
    """Process that runs sign detection using YOLOv8.

    Subscribes to camera frames and publishes detected sign labels
    to the message bus for consumption by the state machine and dashboard.

    Args:
        queueList: Dictionary of multiprocessing queues.
        logging: Logger instance.
        ready_event: Event to signal when threads are ready.
        debugging: Enable debug output.
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processSignDetection, self).__init__(self.queuesList, ready_event)

    def _init_threads(self):
        """Create the sign detection thread."""
        signTh = threadSignDetection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(signTh)
