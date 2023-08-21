import posix_ipc
import mmap
import struct
import os
import psutil
import multiprocessing as mp

class SharedSemaphore:
    def __init__(self, name, initial_value=1):
        self.sem_name = name + f"_{self._get_root_parent_pid()}"
        self.shm_name = name + "_shm"
        self.initial_value = initial_value
        self.acquired = False

        # Create or open semaphore
        try:
            self.semaphore = posix_ipc.Semaphore(self.sem_name, flags=posix_ipc.O_CREAT, initial_value=initial_value)
        except posix_ipc.ExistentialError:
            self.semaphore = posix_ipc.Semaphore(self.sem_name)

        # Create or open shared memory
        flags = posix_ipc.O_CREAT
        opened_new = False
        try:
            opened_new = True
            self.memory = posix_ipc.SharedMemory(self.shm_name, flags=flags, size=4)
        except posix_ipc.ExistentialError:
            self.memory = posix_ipc.SharedMemory(self.shm_name)

        # Map the shared memory
        self.map = mmap.mmap(self.memory.fd, self.memory.size)
        self.memory.close_fd()

        # If the memory was just created, initialize it
        if opened_new:
            if self.map.size() == 0:
                print("RESETTING COUNT!")
                self.set_count(initial_value)

    def _get_root_parent_pid(self):
        pid = os.getpid()
        ppid = os.getppid()
        while ppid != 1:
            parent_process = psutil.Process(pid)
            ppid = parent_process.ppid()  # Get parent PID of the given PID
            if ppid == 1:
                return pid

            pid = ppid

        return pid

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def acquire(self):
        self.semaphore.acquire()
        print(f'Semaphore count is: {self.get_count()}')
        self.acquired = True
        return self

    def release(self):
        if self.acquired:
            self.semaphore.release()
            self.acquired = False

    def get_count(self):
        self.map.seek(0)
        return struct.unpack('i', self.map.read(4))[0]

    def set_count(self, value):
        self.map.seek(0)
        self.map.write(struct.pack('i', value))

    def update_shared_count(self, delta):
        current_count = self.get_count()
        new_count = current_count + delta
        self.set_count(new_count)

    def close(self):
        self.semaphore.release()
        self.map.close()

    def unlink(self):
        posix_ipc.unlink_semaphore(self.sem_name)
        posix_ipc.unlink_shared_memory(self.shm_name)

    def reset(self):
        # Unlink the existing semaphore and shared memory
        posix_ipc.unlink_semaphore(self.sem_name)
        posix_ipc.unlink_shared_memory(self.shm_name)

        # Recreate the semaphore with the initial value
        self.semaphore = posix_ipc.Semaphore(self.sem_name, flags=posix_ipc.O_CREAT, initial_value=self.initial_value)

        # Recreate the shared memory and map it
        self.memory = posix_ipc.SharedMemory(self.shm_name, flags=posix_ipc.O_CREAT, size=4)
        self.map = mmap.mmap(self.memory.fd, self.memory.size)
        self.memory.close_fd()

        # Set the shared memory count to the initial value
        self.set_count(self.initial_value)

        self.acquired = False



import time
import random
import threading

def test_shared_semaphore():
    def worker(id):
        with SharedSemaphore("test_semaphore", initial_value=3):
            print(f"Worker {id} acquired the semaphore.")
            sleep_time = random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
            print(f"Worker {id} released the semaphore after sleeping for {sleep_time:.2f} seconds.")


    threads = []
    for i in range(5):
        thread = mp.Process(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    test_shared_semaphore()