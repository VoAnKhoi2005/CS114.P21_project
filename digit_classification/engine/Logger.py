import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout            # Save the original stdout (console)
        self.log = open(filename, "a")        # Open log file in append mode

    def write(self, message):
        self.terminal.write(message)          # Write message to console
        self.log.write(message)                # Write message to log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()