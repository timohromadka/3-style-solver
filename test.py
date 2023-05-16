import time
from cube import Cube, Cubie 

def timer_decorator(message):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time elapsed for {message}: {elapsed_time} seconds")
            if elapsed_time > 0:  # avoid division by zero
                rate = len(result) / elapsed_time
                print(f"Theoretical rate for {message}: {rate} per second")
            return result
        return wrapper
    return actual_decorator

class CubeTester:
    @timer_decorator("Getting commutators")
    def get_commutators(self):
        return Cube.get_commutators(Cubie.BOTH, degree=1)

    @timer_decorator("Checking commutators")
    def check_commutators(self, comms):
        correct_comms = []
        for comm in comms:
            if Cube.check_pass([[3, 23, 14]], comm, naive=True):
                correct_comms.append(comm)
        return comms

if __name__ == "__main__":
    tester = CubeTester()
    comms = tester.get_commutators()
    correct_comms = tester.check_commutators(comms)
