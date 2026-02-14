from src.templates.workerprocess import WorkerProcess

class processSemaphores(WorkerProcess):
    """Mock process for Semaphores."""
    def __init__(self, queueList, logging, ready_event, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        super(processSemaphores, self).__init__(self.queuesList, ready_event)

    def _init_threads(self):
        """Initialize threads (mock)."""
        pass

    def process_work(self):
        pass
