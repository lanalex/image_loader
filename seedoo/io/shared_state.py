import os
import mmap
import struct

class SharedMemoryFile:
    def __init__(self, initial_value=0, size=4):
        # Determine the root parent process ID
        root_pid = self._get_root_parent_pid()

        # Create a unique filename based on the root parent PID
        self.filename = f"/tmp/shared_memory_{root_pid}"

        # Create or open the shared memory file
        self.file = open(self.filename, 'a+b')

        # Set the file size
        self.file.seek(size - 1)
        self.file.write(b'\0')
        self.file.flush()

        # Memory map the file
        self.map = mmap.mmap(self.file.fileno(), size)

        # If the file was just created, initialize it
        if os.path.getsize(self.filename) == size:
            self.set_value(initial_value)

    def _get_root_parent_pid(self):
        pid = os.getpid()
        ppid = os.getppid()
        while ppid != 1:
            pid = ppid
            ppid = os.getppid(pid)
        return pid

    def get_value(self):
        self.map.seek(0)
        return struct.unpack('i', self.map.read(4))[0]

    def set_value(self, value):
        self.map.seek(0)
        self.map.write(struct.pack('i', value))

    def close(self):
        self.map.close()
        self.file.close()

    def unlink(self):
        os.remove(self.filename)
