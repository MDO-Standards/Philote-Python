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
Tests for per-job state isolation.

The bug these exist to prevent: a server used to share one discipline instance
across every client, so a second client's ``Setup`` rewrote the variable
metadata the first was still using.
"""
import threading
import time
import unittest
from concurrent import futures
from unittest.mock import Mock

import grpc
import numpy as np
from scipy.optimize import rosen

import philote_mdo.general as pmdo
import philote_mdo.generated.data_pb2 as data
from philote_mdo.examples import Paraboloid, Rosenbrock
from philote_mdo.general import Discipline, ExplicitDiscipline
from philote_mdo.general.discipline_server import DisciplineServer
from philote_mdo.general.job import JOB_METADATA_KEY, JobState, JobStore
from philote_mdo.utils.validation import (
    JobCapacityError,
    JobNotFoundError,
    JobStateError,
    PhiloteJobError,
)

from conftest import Aborted, aborting_job_context, job_context, make_server


def serve(discipline_factory, server_cls=pmdo.ExplicitServer, workers=16, **kwargs):
    """
    Starts a real gRPC server on an ephemeral port.

    Returns
    -------
    tuple
        ``(grpc_server, port)``. Stop the server with ``.stop(0)``.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    server_cls(discipline_factory=discipline_factory, **kwargs).attach_to_server(server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    return server, port


class TestJobIsolation(unittest.TestCase):
    """Two clients against one server must not affect each other."""

    def test_concurrent_clients_with_different_shapes(self):
        """
        The bug from issue #76, reproduced exactly.

        Rosenbrock takes its variable shape from an option, so before jobs
        existed whichever client called SetOptions last fixed the shapes both
        clients got, and the other silently received a zero-padded result.
        """
        server, port = serve(Rosenbrock)
        self.addCleanup(server.stop, 0)

        def client_for(dimension):
            client = pmdo.ExplicitClient(
                channel=grpc.insecure_channel(f"localhost:{port}")
            )
            client.send_options({"dimension": dimension})
            return client

        a, b = client_for(2), client_for(10)

        # interleaved exactly as in the issue: A sets up and reads its
        # metadata, then B sets up, then both compute
        a.run_setup()
        a.get_variable_definitions()
        b.run_setup()
        b.get_variable_definitions()

        xa = np.array([1.5, 0.5])
        xb = np.arange(1.0, 11.0)

        self.assertAlmostEqual(a.run_compute({"x": xa})["f"][0], rosen(xa))
        self.assertAlmostEqual(b.run_compute({"x": xb})["f"][0], rosen(xb))

        # and A is still correct after B has been all the way through setup
        self.assertAlmostEqual(a.run_compute({"x": xa})["f"][0], rosen(xa))

        self.assertNotEqual(a.job_id, b.job_id)

    def test_options_do_not_leak_between_jobs(self):
        """A job's options are invisible to another job."""
        server, port = serve(Rosenbrock)
        self.addCleanup(server.stop, 0)

        a = pmdo.ExplicitClient(channel=grpc.insecure_channel(f"localhost:{port}"))
        b = pmdo.ExplicitClient(channel=grpc.insecure_channel(f"localhost:{port}"))

        a.send_options({"dimension": 3})
        b.send_options({"dimension": 7})

        a.run_setup()
        a.get_variable_definitions()
        b.run_setup()
        b.get_variable_definitions()

        shape_of = lambda c: tuple(
            v.shape for v in c._var_meta if v.name == "x"
        )[0]

        self.assertEqual(list(shape_of(a)), [3])
        self.assertEqual(list(shape_of(b)), [7])

    def test_jobs_evaluate_concurrently(self):
        """
        Two jobs may be inside compute() at the same time.

        The discipline here releases the GIL, which is what a compiled solver
        does. A pure-Python discipline is still serialised by the interpreter;
        jobs buy correctness unconditionally and throughput conditionally.
        """
        hold = 0.4

        class Slow(ExplicitDiscipline):
            def setup(self):
                self.add_input("x", shape=(1,))
                self.add_output("f", shape=(1,))

            def compute(self, inputs, outputs):
                time.sleep(hold)
                outputs["f"] = inputs["x"] * 2.0

        server, port = serve(Slow)
        self.addCleanup(server.stop, 0)

        def run(store, index):
            client = pmdo.ExplicitClient(
                channel=grpc.insecure_channel(f"localhost:{port}")
            )
            client.run_setup()
            client.get_variable_definitions()
            store[index] = client.run_compute({"x": np.array([float(index)])})["f"]

        results = [None, None]
        threads = [threading.Thread(target=run, args=(results, i)) for i in range(2)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        self.assertAlmostEqual(results[0][0], 0.0)
        self.assertAlmostEqual(results[1][0], 2.0)

        # serialised would take 2 * hold; overlapped takes about one
        self.assertLess(elapsed, 2 * hold)


class TestJobLifecycle(unittest.TestCase):
    """StartJob, EndJob, KeepAlive and the state machine."""

    def test_client_starts_a_job_lazily(self):
        """An unmodified client script acquires a job on its first call."""
        server, port = serve(Paraboloid)
        self.addCleanup(server.stop, 0)

        client = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )
        self.assertIsNone(client.job_id)

        client.run_setup()

        self.assertIsNotNone(client.job_id)

    def test_describe_rpcs_need_no_job(self):
        """GetInfo and GetAvailableOptions describe the class, not a run."""
        server, port = serve(Rosenbrock)
        self.addCleanup(server.stop, 0)

        client = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )

        client.get_discipline_info()
        client.get_available_options()

        self.assertIsNone(client.job_id)
        self.assertIn("dimension", client.options_list)

    def test_end_job_releases_the_job(self):
        server, port = serve(Paraboloid)
        self.addCleanup(server.stop, 0)

        client = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )
        client.run_setup()
        job_id = client.job_id

        client.end_job()

        self.assertIsNone(client.job_id)

        # the id is genuinely gone from the server
        client._job_id = job_id
        with self.assertRaises(PhiloteJobError):
            client.run_setup()

    def test_job_context_manager_ends_the_job(self):
        server, port = serve(Paraboloid)
        self.addCleanup(server.stop, 0)

        client = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )

        with client.job():
            self.assertIsNotNone(client.job_id)

        self.assertIsNone(client.job_id)

    def test_keep_alive_defers_eviction(self):
        store = JobStore(Discipline, ttl=10.0, sweep_interval=1000.0)
        self.addCleanup(store.close_all)

        job = store.create()
        job.last_used -= 5.0
        stale = job.last_used

        store.get(job.job_id)

        self.assertGreater(job.last_used, stale)

    def test_set_options_after_setup_is_refused(self):
        """
        The metadata was built from the previous values, so a late change
        would leave the job describing itself inconsistently.
        """
        server, job, _ = make_server(DisciplineServer, Rosenbrock)
        context = aborting_job_context(job=job)

        # the legal order: options, then setup
        options = data.DisciplineOptions()
        options.options.update({"dimension": 3})
        server.SetOptions(options, context)

        server.Setup(data.JobHandle(), context)
        self.assertEqual(job.state, JobState.READY)
        self.assertEqual(
            [tuple(v.shape) for v in job.discipline._var_meta if v.name == "x"],
            [(3,)],
        )

        # changing them now would leave the metadata describing the old shape
        later = data.DisciplineOptions()
        later.options.update({"dimension": 4})

        with self.assertRaises(Aborted):
            server.SetOptions(later, context)

        self.assertEqual(
            context.abort.call_args[0][0], grpc.StatusCode.FAILED_PRECONDITION
        )


class TestJobErrors(unittest.TestCase):
    """The failure paths a client has to be able to tell apart."""

    def test_missing_header_is_refused(self):
        server, job, _ = make_server(DisciplineServer, Paraboloid)
        context = aborting_job_context()  # no job id at all

        with self.assertRaises(Aborted):
            server.Setup(data.JobHandle(), context)

        self.assertEqual(
            context.abort.call_args[0][0], grpc.StatusCode.FAILED_PRECONDITION
        )

    def test_unknown_job_is_not_found(self):
        server, job, _ = make_server(DisciplineServer, Paraboloid)
        context = aborting_job_context(job_id="does-not-exist")

        with self.assertRaises(Aborted):
            server.Setup(data.JobHandle(), context)

        self.assertEqual(
            context.abort.call_args[0][0], grpc.StatusCode.NOT_FOUND
        )

    def test_client_raises_rather_than_restarting_silently(self):
        """
        A client that quietly started a replacement job would run the
        optimizer against a fresh discipline and return plausible but wrong
        results, so an unknown job has to be terminal.
        """
        server, port = serve(Paraboloid)
        self.addCleanup(server.stop, 0)

        client = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )
        client._job_id = "never-existed"

        with self.assertRaises(PhiloteJobError):
            client.run_setup()

        # and it did not paper over the failure by starting a new one
        self.assertEqual(client.job_id, "never-existed")

    def test_capacity_is_refused_explicitly(self):
        """
        A job can hold a mesh or a solver, so the limit is an explicit refusal
        rather than an out-of-memory failure.
        """
        server, port = serve(Paraboloid, max_jobs=2)
        self.addCleanup(server.stop, 0)

        held = []
        for _ in range(2):
            client = pmdo.ExplicitClient(
                channel=grpc.insecure_channel(f"localhost:{port}")
            )
            client.start_job()
            held.append(client)

        third = pmdo.ExplicitClient(
            channel=grpc.insecure_channel(f"localhost:{port}")
        )

        with self.assertRaises(Exception) as caught:
            third.start_job()

        self.assertIn("maximum", str(caught.exception).lower())

        # ending one frees the slot
        held[0].end_job()
        third.start_job()
        self.assertIsNotNone(third.job_id)


class TestJobStore(unittest.TestCase):
    """The store itself, without a server in the way."""

    def test_rejects_a_non_callable_factory(self):
        with self.assertRaises(TypeError):
            JobStore(Discipline())

    def test_each_job_gets_its_own_discipline(self):
        store = JobStore(Discipline, ttl=None)
        self.addCleanup(store.close_all)

        a, b = store.create(), store.create()

        self.assertIsNot(a.discipline, b.discipline)
        self.assertIs(a.discipline.job, a)
        self.assertIs(b.discipline.job, b)

    def test_describe_never_hands_out_a_job_instance(self):
        store = JobStore(Discipline, ttl=None)
        self.addCleanup(store.close_all)

        job = store.create()

        self.assertIsNot(store.describe(), job.discipline)
        self.assertIs(store.describe(), store.describe())

    def test_capacity_error(self):
        store = JobStore(Discipline, max_jobs=1, ttl=None)
        self.addCleanup(store.close_all)

        store.create()

        with self.assertRaises(JobCapacityError):
            store.create()

    def test_unknown_job(self):
        store = JobStore(Discipline, ttl=None)
        self.addCleanup(store.close_all)

        with self.assertRaises(JobNotFoundError):
            store.get("nope")

    def test_sweep_evicts_and_tears_down(self):
        torn = []

        class Tracked(Discipline):
            def teardown_job(self):
                torn.append(self.job.job_id)

        store = JobStore(Tracked, ttl=0.01, sweep_interval=1000.0)
        self.addCleanup(store.close_all)

        job = store.create()
        job.last_used -= 1.0

        self.assertEqual(store.sweep(), [job.job_id])
        self.assertEqual(torn, [job.job_id])
        self.assertEqual(len(store), 0)

    def test_close_runs_teardown(self):
        torn = []

        class Tracked(Discipline):
            def teardown_job(self):
                torn.append(self.job.job_id)

        store = JobStore(Tracked, ttl=None)
        job = store.create()

        store.close(job.job_id)

        self.assertEqual(torn, [job.job_id])
        self.assertEqual(job.state, JobState.CLOSED)

    def test_expiry_during_create_still_runs_teardown(self):
        """
        A job reclaimed to make room for a new one must still release what it
        held. Reclaiming the slot without tearing down would leak a mesh or a
        live solver whenever create() beat the sweeper to an expired job.
        """
        torn = []

        class Tracked(Discipline):
            def teardown_job(self):
                torn.append(self.job.job_id)

        # no sweeper thread, so create() is the only thing that can reclaim
        store = JobStore(Tracked, max_jobs=1, ttl=0.01, sweep_interval=1000.0)
        self.addCleanup(store.close_all)

        stale = store.create()
        stale.last_used -= 1.0

        fresh = store.create()

        self.assertEqual(torn, [stale.job_id])
        self.assertEqual(len(store), 1)
        self.assertNotEqual(fresh.job_id, stale.job_id)

    def test_failed_construction_frees_the_slot(self):
        calls = []

        def factory():
            calls.append(1)
            raise RuntimeError("mesh missing")

        store = JobStore(factory, max_jobs=1, ttl=None)
        self.addCleanup(store.close_all)

        for _ in range(2):
            with self.assertRaises(RuntimeError):
                store.create()

        # the first failure did not permanently consume the only slot
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(store), 0)


class TestThreadPoolWarning(unittest.TestCase):
    """max_jobs is not the cap that binds; the gRPC thread pool is."""

    def test_warns_when_pool_is_smaller_than_job_cap(self):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        self.addCleanup(grpc_server.stop, 0)

        server = pmdo.ExplicitServer(discipline_factory=Paraboloid, max_jobs=8)

        with self.assertWarns(RuntimeWarning) as caught:
            server.attach_to_server(grpc_server)

        self.assertIn("thread pool", str(caught.warning))

    def test_quiet_when_pool_is_large_enough(self):
        import warnings

        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        self.addCleanup(grpc_server.stop, 0)

        server = pmdo.ExplicitServer(discipline_factory=Paraboloid, max_jobs=8)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            server.attach_to_server(grpc_server)


if __name__ == "__main__":
    unittest.main()
