import time

class benchmark_timer():

    time_vars = {   'start': None,          # Used to measure time
                    'times': dict(),        # Keeps track of total time
                    'ttimes' : dict(),      # Keeps track of number of number of times a time has been taken
                    'time_hist' : dict(),   # Keeps history of the times
                    'current' : None,       # Tracks the current section name
                    'timers' : dict()}      # Implements a component structure.

    def __init__(self):
        self._start = None
        self._times = dict()
        self._ttimes = dict()
        self._time_hist = dict()
        self._current = None
        self._timers = dict()

    def startTiming(self, name = None):
        if name is None:
            name = '000'
        if self._current is None:
            self._current = name
            self._start = time.clock()
        else:
            keyval = self._current
            if keyval not in self._timers.keys():
                self._timers[keyval] = benchmark_timer()
            self._timers[keyval].startTiming(name)

    def endTiming(self, name = None, tm = None):
        t = time.clock()
        if self._start is None:
            return

        if tm is None:
            t = t - self._start
        else:
            t = tm
        if self._current is not None:
            if name is None:
                name = self._current
            if self._current is name:
                if name in self._times:
                    self._times[name] += t
                    self._ttimes[name] += 1
                    self._time_hist[name].append(t)
                else:
                    self._times[name] = t
                    self._ttimes[name] = 1
                    self._time_hist[name] = [t]
                if self._current in self._timers.keys():
                    self._timers[self._current].endTiming()
                self._current = None
                self._start = None
            else:
                if self._current in self._timers:
                    self._timers[self._current].endTiming(name, tm = t)

    def newTiming(self, name = None):
        if self._current in self._timers and self._timers[self._current]._current is not None:
            self._timers[self._current].newTiming(name)
        else:
            self.endTiming()
            self.startTiming(name)


    def getTimes(self,offset = None):
        if offset is None:
            offset = 1
        else:
            offset += 3
        if not self._times:
            print ("No times recorded yet")
        for key in self._times.keys():
                print ("{:{pad}}'{}' took a total time of {:f} with an average of time {:f}".
                        format('#',key, self._times[key],self._times[key]/self._ttimes[key],pad=offset))
                if key in self._timers:
                    self._timers[key].getTimes(offset)

    def getTimeHist(name):
        assert name in self._time_hist, 'No history found for key {}'.format(name)
        return self._time_hist[name]


    def reset():
        self._times.clear()
        self._ttimes.clear()
        self._time_hist.clear()
        self._start = None
