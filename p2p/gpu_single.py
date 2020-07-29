import torch.multiprocessing as multiprocessing
import time

def wait_for_event(proc_ind,e, temp):
    temp = 0
    print ('wait_for_event: starting')
    e.wait()
    print ('wait_for_event: e.is_set()->', e.is_set())

def wait_for_event_timeout(proc_ind,e, t):
    print ('wait_for_event_timeout: starting')
    e.wait(t)
    print ('wait_for_event_timeout: e.is_set()->', e.is_set())


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    e = multiprocessing.Event()
    temp = 10
    w1 = multiprocessing.spawn(wait_for_event,
                                 args=(e,temp), join=False)

    w2 = multiprocessing.spawn(wait_for_event_timeout, 
                                 args=(e, 2), join=False)

    print ('main: waiting before calling Event.set()')
    time.sleep(3)
    e.set()
    print("temp: ", temp)
    print ('main: event is set')
    w1.join()
    w2.join()