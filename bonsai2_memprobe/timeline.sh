#!/bin/bash
# timeline.sh <label> <cmd...>: system memory taken over time (baseline MemAvailable minus now),
# reports peak, steady-state (median of the last third of the run = decode), and peak RSS.
label=$1; shift
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
avail(){ awk '/MemAvailable/{print $2}' /proc/meminfo; }
base=$(avail); : > /tmp/tl.$$
/usr/bin/time -v -o /tmp/tlt.$$ "$@" > /tmp/tlo.$$ 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do echo $(( base - $(avail) )) >> /tmp/tl.$$; sleep 0.5; done
wait $pid; rc=$?
ok=$(grep -cE "^2, 3, 5, 7, 11, 13, 17, 19, 23, 29|Paris" /tmp/tlo.$$)
rss=$(awk -F: '/Maximum resident set size/{gsub(/ /,"",$2); print $2}' /tmp/tlt.$$)
python3 - "$label" $rc $ok $rss /tmp/tl.$$ <<'PY'
import sys, statistics
label, rc, ok, rss, f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
v = [int(x) / 1048576 for x in open(f) if x.strip()]
tail = v[len(v) * 2 // 3:]
print(f"{label:34s} rc={rc} correct={'yes' if ok else 'NO'}  peak={max(v):5.1f} GiB  steady={statistics.median(tail):5.1f} GiB  RSS-peak={rss/1048576:5.1f} GiB  ({len(v)/2:.0f}s)")
PY
rm -f /tmp/tl.$$ /tmp/tlt.$$ /tmp/tlo.$$
