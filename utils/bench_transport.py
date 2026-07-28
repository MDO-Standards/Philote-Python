# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
"""
Compares the unary and streaming compute transports.

The server runs in a separate process so that its work does not contend with
the client for the GIL, which is what a real deployment looks like and what
makes the per-call transport overhead visible.

Three modes are compared:

    stream            the streaming transport at the default chunk size (1000)
    stream-unchunked  the streaming transport with chunking effectively off
    unary             the unary transport

``stream-unchunked`` is the control. The streaming path fragments arrays at
``StreamOptions.num_double``, and some of the measured streaming cost is that
self-inflicted fragmentation rather than anything intrinsic to streaming. If
the control captures most of the difference, the unary transport is not
earning its keep and the cheaper fix is to stop chunking.

Note that for payloads of a megabyte or more, most of the measured time is
protobuf-to-numpy conversion, which both transports pay equally -- the
transport itself accounts for only a couple of milliseconds. The transports
therefore converge as the payload grows, and the unary advantage is a fixed
per-call saving that matters in proportion to how often you call.

Run from the repository root:

    python utils/bench_transport.py
"""
import argparse
import multiprocessing
import statistics
import time
from concurrent import futures

import grpc
import numpy as np

import philote_mdo.general as pmdo


# a chunk size larger than any array used here, which disables fragmentation
UNCHUNKED_NUM_DOUBLE = 1 << 40


def make_discipline(num_vars, num_elements):
    """
    Builds an explicit discipline with the requested variable count and size.

    The compute body is deliberately trivial so that the measurement reflects
    transport cost rather than discipline cost.
    """

    class BenchDiscipline(pmdo.ExplicitDiscipline):
        def setup(self):
            for i in range(num_vars):
                self.add_input("x%d" % i, shape=(num_elements,))
                self.add_output("y%d" % i, shape=(num_elements,))

        def setup_partials(self):
            pass

        def compute(self, inputs, outputs):
            for i in range(num_vars):
                outputs["y%d" % i] = inputs["x%d" % i] * 2.0

    return BenchDiscipline()


def serve(port, num_vars, num_elements, ready):
    """
    Runs the benchmark server until the parent process terminates it.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    discipline = make_discipline(num_vars, num_elements)
    pmdo.ExplicitServer(discipline=discipline).attach_to_server(server)
    server.add_insecure_port("[::]:%d" % port)
    server.start()
    ready.set()
    server.wait_for_termination()


def make_client(port, mode):
    """
    Builds a client configured for one of the three transport modes.
    """
    client = pmdo.ExplicitClient(channel=grpc.insecure_channel("localhost:%d" % port))
    client.get_discipline_info()
    client.run_setup()
    client.get_variable_definitions()
    client.get_partials_definitions()

    if mode == "unary":
        client.transport = "unary"
        # the pin forces the transport, but raise the threshold too so that
        # the numbers reported here are not confused by a fallback
        client.unary_max_bytes = 64 * 1024 * 1024
    else:
        client.transport = "stream"

    if mode == "stream-unchunked":
        client._stream_options.num_double = UNCHUNKED_NUM_DOUBLE
        client.send_stream_options()
    else:
        client.send_stream_options()

    return client


def time_calls(client, inputs, iterations):
    """
    Returns the per-call wall-clock times, in microseconds.
    """
    samples = []

    for _ in range(iterations):
        start = time.perf_counter()
        client.run_compute(inputs)
        samples.append((time.perf_counter() - start) * 1e6)

    return samples


def run_case(port, mode, num_vars, num_elements, iterations, warmup, threads):
    """
    Runs one configuration and returns its timing summary.
    """
    inputs = {
        "x%d" % i: np.ones(num_elements) for i in range(num_vars)
    }

    clients = [make_client(port, mode) for _ in range(threads)]

    # warm up so that channel setup and the first-call negotiation are not
    # folded into the measurement
    for client in clients:
        time_calls(client, inputs, warmup)

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    if threads == 1:
        samples = time_calls(clients[0], inputs, iterations)
    else:
        with futures.ThreadPoolExecutor(max_workers=threads) as pool:
            results = [
                pool.submit(time_calls, client, inputs, iterations)
                for client in clients
            ]
            samples = [s for r in results for s in r.result()]

    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start

    samples.sort()

    return {
        "median": statistics.median(samples),
        "p99": samples[min(len(samples) - 1, int(0.99 * len(samples)))],
        "client_cpu_us": cpu / len(samples) * 1e6,
        "throughput": len(samples) / wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--port", type=int, default=50081)
    args = parser.parse_args()

    # (variables, elements per variable, concurrent clients)
    cases = [
        (2, 1, 1),
        (2, 1, 4),
        (2, 1, 16),
        (10, 1, 1),
        (100, 1, 1),
        (2, 1000, 1),
        (2, 10000, 1),
        (2, 30000, 1),
        (2, 100000, 1),
        (1, 1000000, 1),
    ]
    modes = ["stream", "stream-unchunked", "unary"]

    print(
        "%-26s %-18s %10s %10s %10s %12s"
        % ("case", "mode", "med (us)", "p99 (us)", "cpu (us)", "calls/s")
    )
    print("-" * 92)

    for num_vars, num_elements, threads in cases:
        label = "%dvar x %d, %dthr" % (num_vars, num_elements, threads)
        baseline = None
        port = args.port
        args.port += 1

        ready = multiprocessing.Event()
        proc = multiprocessing.Process(
            target=serve, args=(port, num_vars, num_elements, ready)
        )
        proc.start()
        ready.wait(timeout=30)

        try:
            # large payloads do not need as many samples to be stable, and the
            # run would otherwise take minutes
            iterations = args.iterations
            if num_vars * num_elements > 10000:
                iterations = max(50, args.iterations // 20)

            # a single message carrying the whole payload has to fit inside
            # gRPC's default receive limit, which rules out the unchunked
            # control and the unary transport for the largest cases
            payload = num_vars * num_elements * 8

            for mode in modes:
                if mode != "stream" and payload > 0.9 * 4 * 1024 * 1024:
                    print(
                        "%-26s %-18s %10s %10s %10s %12s"
                        % (label, mode, "-", "-", "-", "over 4 MiB")
                    )
                    label = ""
                    continue

                stats = run_case(
                    port, mode, num_vars, num_elements, iterations, args.warmup, threads
                )

                if baseline is None:
                    baseline = stats["median"]

                delta = " (%+.0f us)" % (stats["median"] - baseline)
                print(
                    "%-26s %-18s %10.1f %10.1f %10.1f %12.0f%s"
                    % (
                        label,
                        mode,
                        stats["median"],
                        stats["p99"],
                        stats["client_cpu_us"],
                        stats["throughput"],
                        "" if mode == "stream" else delta,
                    )
                )
                label = ""
        finally:
            proc.terminate()
            proc.join()

    print()
    print(
        "Deltas are against the 'stream' row of the same case. If"
        " 'stream-unchunked'\ncaptures most of the 'unary' gain, prefer"
        " raising the chunk size over adding\na second transport."
    )


if __name__ == "__main__":
    main()
