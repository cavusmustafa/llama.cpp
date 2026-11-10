#!/bin/bash
# memprobe.sh <label> <cmd...>
# Peak system-wide memory taken by a run on an iGPU box: baseline MemAvailable minus the
# minimum seen while the command runs (captures GPU/USM allocations that RSS misses),
# plus the process tree's peak RSS from /usr/bin/time.
label=$1; shift
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
avail(){ awk '/MemAvailable/{print $2}' /proc/meminfo; }
base=$(avail); min=$base
/usr/bin/time -v -o /tmp/mp_time.$$ "$@" > /tmp/mp_out.$$ 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do a=$(avail); [ "$a" -lt "$min" ] && min=$a; sleep 0.25; done
wait $pid; rc=$?
rss=$(awk -F: '/Maximum resident set size/{gsub(/ /,"",$2); print $2}' /tmp/mp_time.$$)
printf "%-44s rc=%d  system-peak=%5.1f GiB  process-RSS-peak=%5.1f GiB\n" "$label" $rc \
  $(echo "($base-$min)/1048576" | bc -l) $(echo "$rss/1048576" | bc -l)
rm -f /tmp/mp_time.$$ /tmp/mp_out.$$
