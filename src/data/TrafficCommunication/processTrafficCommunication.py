from src.templates.workerprocess import WorkerProcess

class processTrafficCommunication(WorkerProcess):
    """Mock process for Traffic Communication."""
    def __init__(self, queueList, logging, deviceID, ready_event, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        super(processTrafficCommunication, self).__init__(self.queuesList, ready_event)

    def _init_threads(self):
        """Initialize threads (mock)."""
        pass

    def process_work(self):
        pass
