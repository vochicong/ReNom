import time

time_vars = {   'start': None,          # Used to measure time
                'times': dict(),        # Keeps track of total time
                'ttimes' : dict(),      # Keeps track of number of number of times a time has been taken
                'time_hist' : dict(),   # Keeps history of the times
                'timers' : []}          # Used for component design to keep track of subtimes

def startTiming(name = None):
    if name is not None:
        time_vars['current'] = name
    else:
        time_vars['current'] = None
    time_vars['start'] = time.clock()

def endTiming(name = None):
    t = time.clock()
    assert time_vars['start'] is not None, 'Must call startTiming at least once before endTiming'
    t = t - time_vars['start']
    if time_vars['current'] is not None:
        keyval = time_vars['current']
    else:
        keyval = '000'
    if keyval in time_vars['times']:
        time_vars['times'][keyval] += t
        time_vars['ttimes'][keyval] += 1
        time_vars['time_hist'][keyval].append(t)
    else:
        time_vars['times'][keyval] = t
        time_vars['ttimes'][keyval] = 1
        time_vars['time_hist'][keyval] = [t]
    time_vars['current'] = None
    time_vars['start'] = None

def newTiming(name = None):
    try:
        endTiming()
    except:
        pass
    startTiming(name)

def getTimes():
    if not time_vars['times']:
        print ("No times recorded yet")
    for key in time_vars['times'].keys():
            print ("'{}' took a total time of {:f} with an average of time {:f}".format(key, time_vars['times'][key],time_vars['times'][key]/time_vars['ttimes'][key] ))

def getTimeHist(name):
    assert name in time_vars['time_hist'], 'No history found for key {}'.format(name)
    return time_vars['time_hist'][name]


def reset():
    time_vars['times'].clear()
    time_vars['ttimes'].clear()
    time_vars['time_hist'].clear()
    time_vars['start'] = None
